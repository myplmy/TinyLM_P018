# 5. 학습 속도

벽시계 시간 단축. 현재 m100 300M ≈ dense 97.5분 / tied ~90분.
**주의: 이 스텝은 거의 연산 바운드**(이론 하한 ~2.2s vs 실측 ~2.6~3.3s)라 큰 폭 단축은 어렵다.

| 기법 | 상태 | 버전 | 작동원리(개선 기여) | 트레이드오프 | 특기 |
|---|---|---|---|---|---|
| **bf16 autocast** | ✅ | v4~ | 텐서코어 활용, GradScaler 불필요 | — | 주 matmul이 bf16이라 TF32 이득은 제한적 |
| **torch.compile** | ✅ | v5~ | 그래프 컴파일·커널 융합 | 첫 스텝 컴파일 수 분 | "Not enough SMs"로 max_autotune GEMM은 생략 |
| **pin_memory + non_blocking** | ✅ | v4~ | H2D 전송을 연산과 겹침 | — | Loader가 이미 적용 |
| **TF32 + cudnn.benchmark** | 🧪 | v6 | fp32 matmul→TF32, 고정 shape 오토튠 | 소폭(주연산 bf16이라) | `set_float32_matmul_precision('high')` |
| **reduce-overhead (CUDA그래프)** | 🧪 | v6 (`--compile-mode`) | 커널 런치·파이썬 오버헤드를 그래프로 묶음 → 10~25% 가능 | grad accum과 충돌 → 각 forward 전 `cudagraph_mark_step_begin()` 필요(v6에서 처리), VRAM 약간↑ | 연산 바운드라 이득 불확실. 문제 시 `--compile-mode default` |
| **gradient checkpointing off (`--no-ckpt`)** | ✅**측정·채택** | v4~ | 활성 재계산 제거 → forward 1회 | VRAM **13.5GB**(스필벽 바로 아래). **tied+KD+dense교사에는 금지**(15.7GB↑, OOM 위험) | ★결과 007: 정상상태 **2982→2467 ms/step = -17.3%**. **단일 최대 엔지니어링 레버.** dense 런 표준 |
| **micro-batch 크기 `M = mb×seq`** | 🧪**조건부**(폐기 철회) | — | micro-batch 당 토큰 M 이 클수록 GEMM·런치 효율↑ | **M 무릎 ≈ 8,192.** 아래면 손해, 위면 포화. VRAM 도 M 으로 결정 | ★결과 007+P021B: M=4096 **2550** / M=8192 **2327~2468** / M=10240 2443(포화) / M=12288 **OOM**. 표준(mb8×seq1024)은 **이미 M=8192** 라 더 올릴 이유 없음 — 007의 "이득 0"은 맞았으나 이유("연산바운드")는 **틀렸다**. ★**seq 를 줄이면 mb 를 2배로 올려 M 유지 필수**(P013 설계조건) |
| **데이터 비동기 프리페치** | 💡 | — | 데이터 로딩과 연산 겹침 | 복잡도↑ | 연산 바운드라 이득 미미 → 보류 |
| **Sequence packing (패딩 제거)** | ✅ | v4~ | 문서를 이어붙인 스트림에서 연속 크롭 → 패딩 0, 밀도 100% | — | 이미 적용(Loader). 제안의 "15~30% 이득"은 이미 반영됨 |
| **Muon 옵티마이저** | 💡 | — | Linear 가중치에 스펙트럴(Newton-Schulz) 업데이트 → 같은 loss까지 step↓ | **대배치에서 이득 집중**(소배치·단일GPU엔 제한적), ternary STE와 상호작용 미검증, 구현·튜닝 비용 | muP와 결합 시 HP 전이. 1.7B·대배치 확장 때 유력. 근거 arXiv:2505.02222 |
| **MTP(학습 aux head)** | 💡 | — | 미래 2~4토큰 동시 예측 aux loss → 수렴·표현력↑ | 헤드·loss 추가 연산, 소형 모델 이득 불확실 | factorized 헤드와 결합 가능. 학습용은 DeepSeek-V3식. FastMTP는 추론용(별건) |
| **커스텀 삼진 커널(분리 모듈)** | 🧪→⚠️GPU | v6 (`--ternary-kernel[-triton]`) | int8 codes+그룹alpha 패킹, STE backward 보존. 레퍼런스=기존 경로와 등가, Triton=속도 시도 | GPU 학습 가속은 **원리상 불가**(dequant 후 dense와 동일 FLOPs). `--compile` 병용 시 dynamo 크래시 → 현재 코드가 SystemExit 차단 | **"10×+ 느림"은 정정됨**(결과 003): torch.compile 재컴파일 아티팩트였고, `--compile` 없이 재측정하니 k_triton 160~250ms/step 로 정상. **커널 자체는 문제 없음.** 다만 GPU 학습 목적은 여전히 무의미 → **실사용처 = CPU LUT/AVX2 배포**(P016 3:4 정합). 벤치는 `--compile` 없이 |
| **Skip-Forward / Dynamic KD** | ✅**측정·기본값** | v6 (`--kd-every K [--kd-dynamic]`) | 교사 forward를 K스텝마다 1회 → 교사 연산 1/K. dynamic은 초반 촘촘·후반 성김 | 건너뛴 스텝 KD 신호 0 → K 크면 격차 재확대 | ★결과 005: **정적 k4 가 품질·시간 모두 최적**(3.7875 / 141.0분, full KD 대비 **-0.042·-15%**). **dynamic 우선권장 철회**(150M 혼입판의 dyn4 우세는 역전됨). **KD 기본값 = `--kd-every 4`** |
| **압축 교사 distill** | ✅**측정** | v6 (`--kd-teacher-tag`) | 교사를 dense(132.5M) 대신 압축 tied(≈63.5M)로 → 교사 forward 대폭↓ | 교사 상한=압축 교사 품질, 순환성(교사 1회 학습 비용은 지불) | ★결과 007: k4+압축교사+no-ckpt 조합으로 **교사 오버헤드 = dense-nockpt 대비 +19%**(2940 vs 2467 ms/step), full-KD 대비 스텝시간 **-32%**. `load_dense`가 cfg로 범용 복원. P018 |
| **hidden(E=256) 정확 오프라인 KD** | 💡 | — | 교사 pre-head hidden h_E(256차)만 캐시 → logits=h_E@emb^T 로 **정확 복원**(top-k 손실 없음). 교사 body forward 제거 | 300M×256 fp16=154GB(int8 77GB, int4 38GB) 디스크. 복원 시 V×E matmul | P015 top-k 실패의 대안. E=256은 head 랭크(정확성 하한). 축소=양자화/PQ/랭크절단 |
| **Fused Cross-Entropy 커널** | 💡 | — | 큰 vocab 로짓 materialize 없이 linear+CE 융합 → 메모리·오버헤드↓ | Windows 호환 불확실 | torch.compile이 일부 융합. vocab 32k라 이득 소폭 |
| **FP8 텐서코어 학습** | 🚪**0단계 통과·1단계 조건부** | — (P022) | Ada FP8(E4M3) GEMM = bf16 대비 구조적 2:1. 삼진 dequant 후 GEMM 을 FP8 로 | ★**GEMM 은 스텝의 50%뿐** → 상한 -23%, 캐스팅 오버헤드 감안 **실제 -15% 내외**. activation 양자화가 삼진 위에 겹침(결과 008 의 초선형 악화 우려). `--compile` 상호작용 미검증. 커스텀 autograd(fwd/dgrad/wgrad 3 GEMM) 공수 큼 | ★결과 010: 실측 **1.63~2.08×**(gate/up 1.63, down 2.00, attn 1.86). Ada 2:1 비율과 일치 → **"소형GEMM 무의미" 가설 기각**. 환경 OK(torch 2.10/CUDA 13/sm_89/`_scaled_mm`). `torchao` 불필요 — `torch._scaled_mm` 직접. **1단계 선결 = σ 실측 + REVIEW1 아키텍처 확정** |
| **int8×int8 텐서코어 GEMM** | ⛔검토후보류 | — | activation까지 int8화한 IMMA GEMM | 소형모델 이득 marginal + 양자화오버헤드 + 동적범위손실 + STE복잡. 학습엔 부적합. Ada는 2×(A100/H100의 2~4× 아님) | 결론: 학습 미도입. CPU 배포는 삼진-weight LUT 경로가 별개 |
| **Sophia 옵티마이저** | ⏸**보류** | — (P023) | 대각 Hessian 곡률로 스텝수↓. 논문 125M–1.5B에서 ~2× | 재현성 데이터·튜닝 의존, 삼진 STE 상호작용 미검증, HP 재탐색 | **P026 통과 후 착수**(2026-07-30 결정). P026 과 같은 자원(steps)을 노려 교락되고, 판정에 노이즈 실측이 선결. arXiv:2305.14342 |
| **점진적 스태킹(model growth)** | 💡 | — (P024) | 얕게→깊이 성장, 연산 재사용 → 같은 품질 총 벽시계↓(RAPTR 33%) | 성장 스케줄 민감, CLA/KV-bank 재배선, 어닐 상호작용 | dense 교사 학습비 절감에 적용. arXiv:2402.05913 |
| **어닐 형태(선형 램프 vs 계단)** | 🧪**구현완료·실측대기**(P035) | v6 (`--anneal-shape`·`--anneal-start`) | 계단 어닐로 'FP→별도 QAT' 중복을 **인위 생성**해 논문 기전의 존재 여부를 검정 | 기본값 `linear`/자동 = **종전 동작 무변** | 결과 015 가 남긴 미해결 질문. `qb_wsd60`·`qb_wsd80` 을 대조군으로 재사용해 **2런(3.2h)만으로 2×2** 완성. 배치 `run_P035_anneal.bat`(작성완료) |
| **cooldown-QAT 융합 스케줄(정렬)** | ⏸**실측완료·귀속 미확정**(결과 015) | v6 (`--anneal-end`·`--decay-frac`) | LR 감쇠와 QAT(어닐)를 겹쳐 중복 full-precision 업데이트 제거 | **정렬 고유 효과가 검출되지 않았다**(−0.0128 = 1.1σ < 2σ) | ★결과 015: `qb_wsd80 − qb_wsd60 = −0.0128`. 15% 스텝 절감(`qb_wsd80_s85` vs `qb_cos` **+0.0050**)은 성립하나 **wsd 성분과 분리 안 됨** → 빠진 칸 `qb_wsd60_s85`(P026 단계5) 전까지 채택 보류. arXiv:2509.22935 |
| **WSD 스케줄(스텝 절감 레버)** | ✅**측정·채택 후보** | v6 (`--sched wsd`) | 긴 plateau + 마지막 `decay_frac` 감쇠 | 표준조건 개정은 P026 단계5 후 | ★결과 015: 동일 스텝·동일 종료 LR(peak×0.1)에서 cosine 대비 **−0.0755(6.3σ)**. 시간 비용 0(정상상태 ms/step 2502~2520, 스케줄 무관). **dense 에서만 확인** — tied+KD 미검증 |
| **seq-length warmup(P013)** | 💡→📌**설계조건 확보** | — (P013) | 초반 짧은 seq 로 어텐션 O(seq²) 절감 | **M 유지 필수** — seq 절반이면 mb 2배 | ★P021B: mb16×seq512(M=8192) 가 표준 mb8×seq1024(M=8192) 대비 **-5.7%**. 단 seq·mb 가 동시에 변해 기여도 분리는 안 됨. 이 값을 이득 상한으로 |
| **데이터 풀 다양성(`--pool-tokens`)** | ✅**측정·채택** | v6 | 학습토큰은 그대로 두고 샘플 풀만 확대 → 반복 노출 감소 | 토큰화 1회 비용·디스크 | ★결과 006: 같은 300M 학습에서 풀 300M→600M 만으로 dense **3.8241→3.7045(-0.12)**. **학습시간 증가 0 의 무료 품질 레버.** 신규 기준선은 풀 ≥ 2× 학습토큰 필수. 포화점 미측정 |
| **2:4 준정형 희소(GPU)** | 💡→🧪게이트 | — (P025) | 희소 텐서코어로 학습 GEMM 최대 2×(Ada 지원). 메모리+속도 동시 | ★**"50% 강제-0 리스크" 재평가 필요**(결과 016 §7.2): 우리 표준 TWN 은 **이미 43~46% 가 0** 이라 2:4(비영 50%)는 **TWN 과 거의 같은 희소도**다. 리스크는 희소도가 아니라 **블록 제약이 큰 가중치를 죽이는 것**. 소형GEMM 이득 불확실, cuSPARSELt/Windows 제약 | 마이크로벤치 게이트. 삼진 정합(Sparse-BitNet arXiv:2603.05168). **Arenas(P036) 가 N:M 제약 일반에 적용되므로 P036 결과를 먼저 본다** |
| **URL/메타데이터 prepend(데이터)** | 💡 | — (P012 추가) | 문서 앞 출처/메타 prepend로 목표loss 토큰 30~40%↓ | 한국어 셋은 URL 원본 부재 가능(score 대체) | 연산바운드에서 절대 벽시계 줄이는 데이터측 레버. arXiv:2511.21613 |
| **SplitK 융합 dequant+GEMM(추론)** | 📄참고 | — | 스키니(M=1-16) 메모리바운드 W4A16 GEMM을 SplitK atomic으로 가속 | **추론 decode 전용**. 학습은 M=8192 대배치=연산바운드라 무관. cuBLAS 아닌 naive Triton DP 대비 수치 | 우리 학습엔 부적합. GPU decode 시 ternary_kernel 업그레이드 경로. arXiv:2402.00025 |

## accum/steps 착시 주의 (중요)

이 코드에서 **1 step = accum micro-batch 묶음**이라 ms/step은 accum에 ~선형이다. accum을 줄이면
ms/step은 주지만 step당 토큰(=micro_bs×accum×seq)도 줄어 **고정 토큰까지 총 벽시계는 거의 불변**
(연산 바운드). 즉 "accum 조정으로 step 반감"은 총시간 이득이 아니다. 실제 벽시계 레버는:
**(a) `--no-ckpt`**(재계산 제거, 단일 최대, **실측 -17.3%**·VRAM 확인),
**(b) `M = micro_bs×seq` 를 8,192 이상으로 유지**(무릎. 표준은 이미 충족 — 아래 §M 규칙),
**(c) KD 오버헤드 소거**(dyn4+압축교사), **(d) 데이터효율 P012**(목표loss까지 토큰↓), **(e) seq-warmup
P013**, **(f) FP8 P022**(선검증). 조합 벤치 = **P021**. 정직한 상한: 연산바운드·이론하한 ~2.2s이라
dense step <1250 반감은 비현실적 — KD는 ≈dense까지, dense는 no-ckpt로 소폭.

## ★M 규칙 — micro-batch 당 토큰 (결과 007 + P021B, 2026-07-30)

**`M = micro_bs × seq`** 하나가 **속도와 VRAM 을 동시에** 결정한다. micro_bs 단독으로 보지 말 것.

| M | 정상상태 ms/step | VRAM(`--no-ckpt`) | 판정 |
|---|---|---|---|
| 4,096 (mb8×seq512) | 2550 | 13.5GB 미만(추정) | **무릎 아래 — -8.7% 손해** |
| **8,192** (mb8×seq1024 / mb16×seq512) | 2468 / 2327 | **13.5GB** | **무릎 = 표준** |
| 10,240 (mb10×seq1024) | 2443(정규화) | 15.2GB | 포화(+1% 이내), 스필 구간 |
| 12,288 (mb12×seq1024 **또는** mb24×seq512) | — | **OOM** | 한계 초과 |

M=12,288 은 **서로 다른 두 형상(mb12×1024, mb24×512)이 똑같이 OOM** 했고 PyTorch 할당량
(12.65 GiB)까지 같았다 → VRAM 이 M 의 함수임이 교차 확인됐다.

**실무 규칙**: seq 를 절반으로 줄이면 micro_bs 를 2배로 올려 M=8,192 를 유지한다(P013 선결조건).
그러지 않으면 8.7% 를 잃는다.

## VRAM/스필 주의 (Windows, RTX 4070 Ti Super 16GB)

- VRAM을 ~15GB까지 채우면 **WDDM 공유메모리 스필**로 PCIe 왕복 → 7배 느려짐.
  목표는 "채우기"가 아니라 **스필 절벽(≈13~14GB) 아래에서 throughput 최대화**.
- `expandable_segments`는 Windows 미지원(경고만) → posix에서만 설정(v6).
- NVIDIA 제어판 "시스템 메모리 폴백 안 함"으로 두면 스필 대신 OOM(한계 파악 용이).

## WSL2 이전 검토 (작업환경)

- **여는 것**: Triton(FP8/커널/2:4·SplitK) 성숙, bitsandbytes 8-bit 옵티마이저, flash-linear-attention
  (P004 KDA), `expandable_segments`(posix) → 위 여러 레버의 전제조건을 해제. 일부 보고는 WSL2가
  네이티브 Windows보다 빠름.
- **안 여는 것(주의)**: WSL2도 GPU가 **WDDM 경유**라 **스필 절벽(≈13~14GB)은 그대로**(WSL issue #10452:
  VRAM 근접 시 공유메모리 스필로 급감). `/mnt/c` 파일 I/O 느림 → 데이터·HF캐시는 WSL2 ext4 안에 둘 것.
- **결론**: 커널레벨(FP8/2:4/ternary kernel)·8-bit 옵티마이저·KDA를 실제 추진하면 WSL2 권장. 순수
  bf16 cuBLAS 학습만이면 이득 제한적(스필 벽 불변). 근거: triton-windows(≥3.3 Windows 지원 존재하나
  Linux가 성숙), bitsandbytes Windows/ WSL2 모두 CUDA 오류 보고 잔존.

## 리서치 근거·링크 (2026-07 조사)

- 옵티마이저: [Sophia (arXiv:2305.14342)](https://arxiv.org/abs/2305.14342) 2× 주장(125M–1.5B),
  [Muon (arXiv:2505.02222)](https://arxiv.org/html/2505.02222v1) 토큰 15%↓·대배치 편중,
  재현성 유보 [Benchmarking Optimizers (arXiv:2509.01440)](https://arxiv.org/pdf/2509.01440).
- 성장/커리큘럼: [Progressive Subnetworks/RAPTR (arXiv:2402.05913)](https://arxiv.org/abs/2402.05913),
  [Curriculum-Guided Layer Scaling (arXiv:2506.11389)](https://arxiv.org/abs/2506.11389).
- 스케줄/QAT: [Compute-Optimal QAT (arXiv:2509.22935)](https://arxiv.org/abs/2509.22935),
  [Sub-100M QAT schedule×bit (arXiv:2605.25966)](https://arxiv.org/pdf/2605.25966).
- 희소/커널: [2:4 Sparsity Pretraining (arXiv:2404.01847)](https://arxiv.org/html/2404.01847v3),
  [PyTorch 2:4 blog](https://pytorch.org/blog/accelerating-neural-network-training/),
  [Sparse-BitNet (arXiv:2603.05168)](https://arxiv.org/html/2603.05168v1),
  [SplitK W4A16 (arXiv:2402.00025)](https://arxiv.org/abs/2402.00025).
- FP8: [TorchAO (arXiv:2507.16099)](https://arxiv.org/pdf/2507.16099),
  [FP8 mechanics](https://r0m1t.com/fp8forllms.html).
- 데이터: [Metadata/URL prepend (arXiv:2511.21613)](https://arxiv.org/pdf/2511.21613).
- 환경: [triton-windows](https://github.com/woct0rdho/triton-windows),
  [WSL GPU 스필 issue #10452](https://github.com/microsoft/WSL/issues/10452).

---

## ★2026-08-06 갱신 — 차단 해제 2건 · 미실험 목록

### 차단 해제

| 기법 | 종전 상태 | **지금** | 근거 |
|---|---|---|---|
| **FP8(P022 단계1)** | ⏸ "σ 실측·REVIEW1 후" | **★차단 해제** | σ=0.012 실측(결과 012) **완료**, REVIEW1 은 결과 024 로 **mC 확정 권고**까지 갔다. 두 선결 조건이 모두 충족됐다. 0단계 게이트는 이미 통과(결과 010, 순수 GEMM 1.63~2.08×) |
| **Sophia(P023)** | ⏸ "P026 통과 후" | **★차단 해제** | P026 은 결과 015 로 **종결**(정렬 무효), P035 로 **계열 종결**(형태도 무효). 자원 교락 대상이 사라졌다 |

> **FP8 의 기대값을 미리 낮춰 둔다**: 결과 010 이 이미 계산했다 — GEMM 이 스텝의 **50%** 뿐이라
> end-to-end 는 **−15% 내외**(상한 −23%)다. "1.63~2.08×" 를 그대로 인용하면 안 된다.

### P014(커스텀 삼진 커널) — ★새로운 근거가 생겼다

종전 판단은 "Triton 커널이 레퍼런스보다 빠르지 않으면 이득 없음" 이었다.
결과 016 §10.4 가 **다른 각도**를 열었다:

> batch 1 디코드에서 세 모델의 tok/s 가 파라미터 2.26배 차이에도 **5% 이내**다.
> **런치 오버헤드 바운드** 가설이 맞다면, 이득은 GEMM 을 빠르게 하는 데서 오는 것이 아니라
> **커널 개수를 줄이는 융합**에서 온다. 커스텀 커널은 정확히 그걸 할 수 있다
> (dequant + GEMM 을 하나로).

**단 가설 검정이 선결**이다 → `run_P030_stage4_layerscaling.bat`(수분, GPU 0).
**층 수를 바꿔도 tok/s 가 안 변하면** 융합도 소용없다.

### 아직 실험하지 않은 학습속도 기법

| 기법 | 상태 | 왜 아직 안 했나 | 다음 조건 |
|---|---|---|---|
| **FP8 학습(P022 단계1)** | **가능** | 선결이 방금 풀렸다 | **배치 작성 완료** `run_P022_stage1_fp8.bat` |
| **★KD 교사에 `freeze_quant`+`drop_latent`** | 💡**최우선(값싸다)** | — (P042) | 교사는 가중치가 **고정**인데 매 스텝 `refresh_quant()` 를 돌고 fp32 latent 도 들고 있다. 추론용으로 **이미 구현된 함수 두 개**를 학습 루프의 교사에 붙이면 된다 | 없음(교사는 backward 가 없다) | ★결과 014: 재양자화가 CPU 추론 시간의 **약 79%** 였다(디바이스는 다르나 같은 연산). 그리고 교사 latent 해제로 **VRAM 이 남으면 tied+KD 에도 `--no-ckpt` 가 열린다 = 추가 −17.3%**. 구현 소, 기대 큼 |
| **FP8(Ada)** | ⏸**조건부 보류** | — (P022) | 텐서코어 FP8 GEMM | 스케일링 팩터 관리 실패 시 **조용한 발산**. 삼진 STE backward 와의 상호작용 미분석 | ★결과 010: GEMM 1.63~2.08× 이나 GEMM 이 스텝의 50% → end-to-end **−15%**(표준 런 18분). **2026-08-06 판단: 지금은 구현 안 한다** — 메모리 목표와 무관하고 위 두 행이 같은 자릿수를 구현 거의 없이 준다. **트리거 = 20 GPU시간 이상 캠페인(P041 단계2·P033) 확정 시** |
| Sophia(P023) | **가능** | 선결이 방금 풀렸다 | HP 재탐색 비용이 크다 — FP8 다음 |
| 데이터 비동기 프리페치 | 보류 | 연산 바운드라 이득 미미 | 학습이 데이터 바운드가 되면 |
| int8×int8 IMMA GEMM(학습) | ⛔기각 | 소형모델 이득 marginal + 동적범위 손실 | 재검토 안 함 |
| CUDA Graphs(P027) | 미착수 | 런치 오버헤드 제거 — **결과 016 §10.4 가 이걸 다시 띄웠다** | P030 단계4 결과 후. **추론 쪽 이득이 더 클 수 있다** |
| 점진적 스태킹(P024) | 미착수 | 교사 학습 가속용. 교사 재학습 계획이 없다 | P033 스케일업 시 |
| 2:4 희소 GPU 가속(P025) | 미착수 | 희소 텐서코어. 3:4 와 별개 | 3:4 결론(결과 019 §9) 이후 우선순위 낮음 |
| **★압축 교사**(`--kd-teacher-tag`) | 🧪**구현됨·미사용** | v6 (P018, B-2) | dense 교사 대신 tied 교사로 KD | ⚠️**교사가 약해져 품질 영향** | ~교사 forward 비용 **절반** + VRAM 확보. 결과 005 가 압축 교사를 이미 측정한 이력 있음 |
| **★오프라인 KD 캐시** | 🧪**구현됨·미사용** | v6 (`--kd-cache`, P015, B-3) | 교사 top-k 를 미리 캐시 → 교사 forward **0** | ⚠️top-k 16 이라 **정보 손실**, 디스크 큼 | ~학생만 돌아 **−35% 추정** |
| **★교사 forward int8**(신규) | 💡 | — (P042 §7, B-1) | 교사도 `to_int8()`. ★결과 016 §12.3: **GPU int8 언팩은 −12~15% 뿐** | 속도 −1%(교사가 1/4 스텝) | ~VRAM **−350MB** → **`--no-ckpt`(−17.3%) 개방 확률↑**. **VRAM 을 −1% 로 사는 셈** |
| ~~accum 조정으로 스텝 반감~~ | 🚫**이득 0 확정** | — (B-5) | — | — | 🚫★결과 007: ms/step 이 accum 에 **~선형** → **총 벽시계 이득 없다.** 다시 시도하지 않는다 |

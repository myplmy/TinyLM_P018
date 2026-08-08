# P018 — 압축 교사 distill (온라인 KD 가속 + 교사 메모리 절감)

## 목적
교사가 **dense 132.5M** 이라 forward가 비싸다(KD ~2×). 이미 dense 품질에 도달한 **압축 tied
체크포인트(예: t_kdinit g4, 72.9M)를 교사로** 쓰면 교사 forward가 ~0.55×로 싸진다. 질문:
**압축 교사로 distill 해도 학생 품질이 dense-교사와 동등한가?**(교사 상한 = 압축 교사 품질)

## 구현 (완료, 바로 실행 가능)
- `init_utils.load_dense` 는 체크포인트의 **cfg로 복원** → tied 교사도 그대로 로드(범용). 교사는
  `set_anneal(1.0)` 로 완전 삼진(배포 모드) forward.
- `cli.py`: `--kd-teacher-tag TAG` → 교사 경로를 `{base}_{TAG}.pt` 로(기본 dense 대신). KD/skip-forward
  플래그와 조합 가능.

## 전제
- 교사로 쓸 압축 모델이 먼저 있어야 함: 예 `t_kdinit`(g4, dense KD로 학습해 dense≈품질). 없으면 먼저 학습.
- 교사 상한 원칙: **압축 교사 품질 ≤ dense** 이면 학생도 그 이하. t_kdinit 이 dense와 동등(+0.001)일 때만 유효.

## 실험 매트릭스 (m100, ko-en, 300M, 학생 g8)
| 태그 | 교사 | 교사 forward 비용 | 기대 |
|---|---|---|---|
| t_kd_g8 (기존) | dense 132.5M | 1.00× | 격차 +0.005 기준 |
| t_kd8_ct | 압축 t_kdinit 72.9M | ~0.55× | 격차 유지 & 시간↓? |
| t_kd8_ct_k2 | 압축 교사 + `--kd-every 2` | ~0.27× | P017 결합 상한 |

판정: `compare`(dense 기준) 격차 ≤ +0.07 유지하며 wall_sec 감소. 압축교사 격차가 dense교사보다
크게 벌어지면 "교사 상한" 확인(압축 교사 자체 품질이 병목).

## 실행 (사용자 대리 수행)
> 토큰 예산: `--accum 16`(유효배치 131K) = 300M. 생략(accum8)하면 150M만 학습되니 주의(P017 005 교훈).
```
python run100m.py train --arch tied --data ko-en --tokens 300M --steps 2289 --micro-bs 8 --accum 16 ^
  --lr 1e-3 --compile --kd --init-from --mlp-group 8 --kd-teacher-tag t_kdinit --tag t_kd8_ct
python run100m.py train --arch tied --data ko-en --tokens 300M --steps 2289 --micro-bs 8 --accum 16 ^
  --lr 1e-3 --compile --kd --init-from --mlp-group 8 --kd-teacher-tag t_kdinit --kd-every 2 --tag t_kd8_ct_k2
python run100m.py compare --tag t_kd8_ct
python run100m.py compare --tag t_kd8_ct_k2
```

### 실행 명령 초안 (순수 ASCII. 착수 시 `run100m_P018.bat` 로 신규 작성할 것)
```
@echo off
REM ===== P018 compressed-teacher distill (teacher = t_kdinit tied ckpt) =====
echo [1/2] t_kd8_ct (compressed teacher, full KD)
python run100m.py train --arch tied --data ko-en --tokens 300M --steps 2289 --micro-bs 8 --accum 16 --lr 1e-3 --compile --kd --init-from --mlp-group 8 --kd-teacher-tag t_kdinit --tag t_kd8_ct
if errorlevel 1 goto ERROR
echo [2/2] t_kd8_ct_k2 (compressed teacher + skip-forward K=2)
python run100m.py train --arch tied --data ko-en --tokens 300M --steps 2289 --micro-bs 8 --accum 16 --lr 1e-3 --compile --kd --init-from --mlp-group 8 --kd-teacher-tag t_kdinit --kd-every 2 --tag t_kd8_ct_k2
if errorlevel 1 goto ERROR
python run100m.py compare --tag t_kd8_ct
python run100m.py compare --tag t_kd8_ct_k2
echo done.
pause
exit /b 0
:ERROR
echo [WARN] stopped: error during run.
pause
```

## 비판/리스크
- **순환성**: 압축 교사도 원래 dense KD로 만든 것 → dense 교사 학습 비용은 이미 한 번 지불. 이득은
  "이후 여러 학생을 값싼 교사로" 재사용할 때. 1회성이면 이득 없음.
- 삼진 교사 forward를 **삼진 커널로 더 빠르게?** → GPU에선 커널이 cuBLAS보다 느림(P014/003 참조).
  따라서 압축 교사도 GPU에선 일반 F.linear(bf16) forward가 최적. 커널 이점은 CPU 배포뿐.

---

# ★2026-08-07 재개 — **전제가 두 군데 틀렸다.** 목적을 다시 쓴다

> 위 본문은 2026-07 작성분이고 **그 사이 우리가 직접 측정한 것들이 전제를 바꿨다.**
> 지우지 않고 남긴다(철회 이력 보존). **아래가 현행 설계다.**

## R1. ⚠️ 틀린 전제 두 개

| # | 원문의 전제 | **우리 측정** | 근거 |
|---|---|---|---|
| **X1** | *"압축 교사면 교사 forward 가 **~0.55×** 로 싸진다"* | 🚫 **거짓.** **타잉은 컴퓨트를 안 줄인다** — `report()` FLOPs 식에 `mlp_group` 이 **없고**, 층 수도 그대로다. 실측 이득은 **−3.15%**(타잉 단독, 캐시 지역성) | `CLAUDE.md` 아키텍처 사실 · 결과 033 |
| **X2** | *"교사 상한: 압축 교사 품질 ≤ dense 이면 학생도 그 이하"* | ⚠️ **전제의 방향이 의심스럽다.** 결과 021 은 공통 val 에서 **`mC` 가 `p6d`(600M 풀 dense)를 이긴다**고 했고, **현행 KD 교사는 `p6d` 가 아니라 300M 풀 `dense`** 다(결과 006: 풀 2× = **−0.12**) | 결과 021 · 006 |

> ★**X2 가 이 실험의 새 동기다.** *"압축했으니 약한 교사"* 가 아니라
> **"지금 쓰는 교사보다 좋을 수도 있는, 그러면서 더 작은 교사"** 일 가능성이 있다.
> ⚠️ 단 **아직 확인 안 됐다** — 두 체크포인트를 **같은 val 셋**에서 재본 적이 없다(함정 2).
> **그래서 단계 R0 가 그것부터 잰다**(GPU 수 분).

## R2. 그러면 무엇을 사는가 — **속도가 아니라 VRAM**

교사를 dense(`m100_ko-en_300M_dense.pt`, 삼진 latent **123.86M**)에서
tied(`mC_wsd`, 유니크 삼진 **54.85M**)로 바꾸면:

| 항목 | dense 교사 | **mC_wsd 교사** | 차 |
|---|---:|---:|---:|
| 삼진 latent(fp32) | 472.5 MiB | **209.3 MiB** | **−263 MiB** |
| dequant 사본(fp32) | 472.5 MiB | **209.3 MiB** | **−263 MiB** |
| **합(교사 상주)** | **945 MiB** | **419 MiB** | ★**약 −0.51 GB** |
| KV 소유 층 | 20/20 | **10/20**(CLA2) | 활성 메모리도 준다 |
| FLOPs/토큰 | 동일 | **동일** | X1 |

★**이건 P042 §7 의 안 B 다.** 계획 P042 는 그 안의 리스크를 *"교사가 약해져 품질 영향"* 이라 적었는데
**R1-X2 가 그 리스크 평가를 뒤집을 수 있다.** 그리고 **P042(교사 latent 해제, 472.5 MiB)와 합치면
약 0.7 GB** 다 — 그래도 `--no-ckpt`(1.7~2.7 GB 필요)에는 못 미친다. **미리 적어 둔다.**

## R3. 질문

| # | 질문 |
|---|---|
| **RQ0** | 현행 dense 교사와 `mC_wsd` 중 **누가 더 좋은가** — 같은 val 셋에서 |
| **RQ1** | `mC_wsd` 를 교사로 KD 한 학생이 dense 교사 학생(`mC_wsd` 자신)과 **동급인가** |
| **RQ2** | 교사 VRAM 약 −0.5 GB 가 실측되는가 |
| **RQ3** | 벽시계가 줄어드는가 — **X1 때문에 기대는 낮다**(−3% 대) |

## R4. ★예측

| # | 예측 | 근거 | 틀리면 |
|---|---|---|---|
| **R-P1** | **RQ0: `mC_wsd` 가 dense 교사를 이긴다**(공통 val 에서 −0.05 이상) | 결과 021(mC ^< p6d) + 결과 006(풀 2× = −0.12). 현행 교사는 300M 풀이다 | 지면 원문 X2 가 맞고 이 실험은 **"교사 상한" 확인**으로 축소된다 |
| **R-P2** | **RQ1: 학생 Δ 는 −0.02 ~ +0.03** — 즉 **동급이거나 약간 낫다** | 교사가 더 좋으면 soft target 도 더 좋다. 단 **자기증류(born-again) 성격**이라 이득이 클 이유도 없다 | +0.05 넘으면 **교사 상한이 아니라 다른 것**(교사 다양성 상실 등)이고 그게 더 흥미롭다 |
| **R-P3** | RQ2: 교사 상주 **약 −0.5 GB** 실측 | 회계(R2) | 안 맞으면 회계가 틀렸다 |
| **R-P4** | RQ3: 벽시계 **−0 ~ −5%** | X1. **"0.55×" 를 인용하면 안 된다** | 크게 빨라지면 X1 이해가 틀린 것 |
| **R-P5** | ⚠️ **`--no-ckpt` 는 여전히 안 열린다** | 0.51 + 0.47 = 0.98 GB ^< 1.7~2.7 GB | 열리면 기준표 §5 의 15.7 GB 가 낡은 것(P042 단계2 와 같은 결론) |

⚠️ **자기증류 순환성 주의**(원문 §비판 그대로 유효): `mC_wsd` 자신이 dense 교사로 만들어졌다.
**교사 학습 비용은 이미 지불됐다** — 이득은 *"이후 여러 학생을 값싼 교사로"* 재사용할 때다.

## R5. 단계 설계 — ✅ **완료 2026-08-08 = [결과 037](../test_result/037_20260808_P018-교사가-학생보다-나빴다-압축교사가-더-좋다.md)**

> **RQ0 ★최대 산출물**: `dense`(현행 KD 교사) 3.8080 vs `mC_wsd` 3.6984 → **paired +0.1096**
> (SE 0.0014, t 78.04, dense 승률 **1.8%**) = **분해능의 4.6배**. **교사가 학생보다 나빴다.**
> ⚠️ 풀 핸디캡(300M vs 600M) 섞임 — *"dense 아키텍처가 나쁘다"* 로 읽지 말 것.
> **RQ1**: `mC_ct` 3.6764 vs `mC_wsd` 3.6984 → paired **−0.0220** = **동급, 방향은 압축 교사 쪽**(R-P2 적중).
> **RQ2**: reserved **10.59 vs 12.47 = −1.88 GiB**(alloc −0.51). **R-P3 을 3.7배 과소예측**했다 —
> 예측은 교사 파라미터만 셌는데 실제로는 **유니크 MLP 20→6 으로 할당 churn 이 줄어** 단편화가 함께 줄었다.
> **RQ3**: 정상상태 3521.2 vs 3474.5/3521.9 = **무변** → **R-P4 적중, X1(0.55× 주장) 실측 반박**.
> **`--no-ckpt`**: 완주는 했으나(14.95 GiB) **스필해서 +26.4% 느렸다**(R-P5 부분 빗나감).
> ⚠️ 배치 안에서는 **`CUBLAS_STATUS_EXECUTION_FAILED`** — 결과 037 §7 에 원인·해결안.
> → **RG0~RG3 전부 통과. 압축 교사는 드롭인이고 P042 §7 안 B 는 '기본' 이 됐다.**

### R5.1 원설계 — 배치 `run_P018_compressed_teacher.bat` (약 4.2h)

| 단계 | 무엇 | 시간 | 게이트 |
|---|---|---|---|
| **R0** | `paired_eval --models dense mC_wsd` — 교사 후보 둘을 **같은 val 셋**에서 | 수 분 | **RQ0 확정.** ⚠️dense 는 300M 풀 학습이라 **풀 핸디캡을 안고 재는 것**이라고 결과문서에 명시 |
| **R1** | 250스텝 × 1, `--kd-teacher-tag mC_wsd` | 20분 | **VRAM 실측**(RQ2)·속도(RQ3). **val 안 읽는다** |
| **R2** | 250스텝 × 1, 위 + `--no-ckpt` | 20분 | **RQ 보너스**: 압축 교사로 `--no-ckpt` 가 열리나. **OOM 이 답** |
| **R3** | 2289스텝 × 1, `--kd-teacher-tag mC_wsd` | 3.5h | **RQ1.** 대조군 `mC_wsd` 재학습 안 함 |
| **R4** | `paired_eval --models mC_wsd mC_ct` | 수 분 | **Δ < 0.024 = 동급** |

★**R3 의 조건은 `--kd-teacher-tag` 하나만 대조군과 다르다.** `--no-ckpt` 는 **R2 에서만** 쓴다
(R3 에 쓰면 grad-ckpt 조건이 달라져 비교가 무효다 — 함정 2).

## R6. 판정

| 게이트 | 통과 | 실패 시 |
|---|---|---|
| **RG0** 교사가 실제로 바뀌었다 | 로그에 `[kd] 교사 로드 완료` + 교사 경로가 `mC_wsd` | 안 바뀌면 **대조군을 재학습한 것** = 3.5h 낭비 |
| **RG1** 품질 | paired Δ < 0.024 | 0.024~0.05 = 교사 상한 존재 / ^> 0.05 = 압축 교사 기각 |
| **RG2** VRAM | 교사분 약 −0.5 GB | 안 맞으면 R2 회계 오류 |
| **RG3** 안정성 | json `grad_max` < 3 | 10 이상 = 판정 불가 |

## R7. 한계

- **자기증류다.** 학생과 교사가 **같은 아키텍처·같은 부모**다. 이득/손해 어느 쪽도 일반화 못 한다.
- **R0 은 풀이 다른 두 체크포인트를 같은 val 에서 비교**한다 — *"어느 쪽이 이 val 을 잘 맞히나"* 는
  유효하지만 *"dense 가 구조적으로 나쁘다"* 로 읽으면 안 된다. **300M 풀 핸디캡이 섞여 있다.**
- 시드 1개. `--kd-every 2` 결합(원문 매트릭스 3행)은 **이번에 안 한다** — 변수 하나씩.
- **X1 때문에 속도 이득 기대는 낮다.** 이 실험을 "KD 가속" 으로 팔면 안 된다. **VRAM 실험**이다.

## R8. preflight

- [ ] 태그 `mC_ct` · `mC_ct250` · `mC_ct250_nockpt` — **신규**, 기준표 §2.6 등록
- [ ] 대조군 `mC_wsd` **재학습 금지**(`--kd-teacher-tag` 하나만 다르다)
- [ ] R3 에 **`--no-ckpt` 금지**(조건 일치)
- [ ] 판정은 `final` + paired. `grad_max` 는 json
- [ ] R0 결과를 **결과문서에 풀 핸디캡과 함께** 적는다


---

## R9. ★단계2 신설 (2026-08-08) — **압축 교사 + `--kd-teacher-infer` 조합**

결과 037 §5 가 남긴 질문: **두 no-ckpt 런의 스필 거동이 정반대였다.**

| | `mC_ct250_nockpt` | `mC_tinf_nockpt` |
|---|---|---|
| reserved | 14.95 GiB | **15.37(더 많다)** |
| 교사 | 압축, **매 forward `refresh_quant()`** | dense, **`freeze_quant()`** |
| 순간 ms/step | **불규칙 +48.5%(p90/p10)** | **±7.4%** |
| 결과 | **+26.4% 느림** | **−20.5% 빠름** |

**가설**: `--kd-teacher-infer` 는 교사의 **할당 패턴을 정적으로** 만든다. 압축 교사는 그것 없이는
**유니크 MLP 6개 + 어텐션의 `_wq` 를 매 교사 forward 마다 새로 만들었다 버린다.**

★**둘을 함께 켠 조합은 한 번도 안 돌렸다.** 계산상:

```
mC_ct250(압축 교사, ckpt ON)           10.59 GiB
  + --kd-teacher-infer (−1.03, 미검증 중복)  ≈  9.6 GiB
  + --no-ckpt          (+3.9~4.4)            ≈ 13.5~14.0 GiB   여유 2.0~2.5 GiB
```

**단계2 설계**(약 40분, 250스텝 × 2):

| 런 | 태그 | 무엇 |
|---|---|---|
| 1 | `mC_cti250` | 압축 교사 + `--kd-teacher-infer`, **ckpt ON** — 중복분 실측 |
| 2 | `mC_cti250_nockpt` | 위 + `--no-ckpt` — **여유 있게 열리는가** |

**게이트**: ① reserved 가 9.4~9.9 GiB 면 두 레버가 **가산**이다 ② `--no-ckpt` 런의
`check_spill.py` 판정이 **"정상"** 이어야 한다(완주만으로는 부족 — 결과 037 이 그 함정을 보여 줬다).

⚠️ **`--no-ckpt` 런은 반드시 단독 실행**하고 **직전 60초 대기**를 둔다(결과 037 §7.3).
⚠️ 스필 정책은 **금지**로 두고 잰다 — 허용 상태로 재면 게이트가 무의미하다
(`docs/methods/09_training_memory.md` §4.3).

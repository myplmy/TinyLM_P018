# 커스텀 삼진 학습 커널 — 준비 문서 (작성 대기)

> 목적: **학습 forward에서 삼진 가중치를 저비트로 직접 연산**해 GEMM 비용을 줄인다.
> 현재는 삼진을 bf16로 dequant한 뒤 **풀 bf16 GEMM**을 돌려, 학습 중 삼진의 연산 이득이 0이다.
> backward는 STE(window)를 보존해야 한다. **이 문서는 준비용 — 커널 작성은 아직 대기.**

> **참고(SplitK, arXiv:2402.00025)**: IBM/Meta의 W4A16 융합 dequant+GEMM은 SplitK atomic 분해로
> **스키니 M=1–16(추론 decode) 메모리바운드** GEMM을 naive Triton DP 대비 65%(A100)~124%(H100)
> 가속한다. **우리 학습(M=micro_bs×seq=8192, 연산바운드)엔 무관** — SplitK 이득은 작은 M에 집중.
> 다만 **GPU decode(generate.py, M=1)** 를 언젠가 가속하려면 현재 naive DP 삼진 커널을 SplitK/StreamK로
> 업그레이드하는 것이 정확한 경로다(P014 결과의 "10× 느림" 원인=naive DP·recompile과 정합). 학습용이 아님.

## 왜 필요한가 (현 병목)
- 현재 `TLinear.forward`: `F.linear(x, wq)` 에서 `wq`는 dequant된 삼진(bf16) → **bf16 GEMM 풀 비용**.
- 100M/300M 학습 ~100~200분의 대부분이 이 GEMM(연산 바운드). 삼진의 **메모리 이점은 배포에만**, 학습 속도엔 미반영.
- 저비트 forward GEMM이면 MLP/어텐션 연산을 줄여 **학습 절대시간**을 낮출 수 있다(스케일 단계에서 결정적).

## 핵심 과제
- **학습은 full-precision latent weight의 gradient가 필요**하다(STE window `1/(1+(aw/clip·alpha)^4)`).
  기존 추론 커널(T-MAC, llama.cpp TQ1_0/TQ2_0)은 **forward 전용**이라 그대로 못 쓴다.
- 따라서 커스텀 = **저비트 forward + STE 보존 backward**. 순전파는 삼진×활성, 역전파는 latent weight로 흐르는 grad에 window.

## 참고 자료 (수집)
| 자료 | 우리에게 쓸 부분 |
|---|---|
| **T-MAC** (LUT 기반 저비트 GEMM, CPU) | 삼진×활성을 사전계산 LUT로. CPU 배포 커널과 통일 가능 |
| **BitBLAS** (GPU mixed-precision GEMM) | GPU 저비트×저정밀 GEMM. 일부 학습 경로 참조 |
| **llama.cpp TQ1_0/TQ2_0** | g128 삼진 레이아웃·패킹 포맷(배포 호환 기준) |
| **DeepGEMM** | SM90/100 전용(우리 SM89 불가). **원칙만**: 그룹 스케일을 커널 1급 시민, M축 grouped GEMM, JIT |
| **Triton** | Windows/SM89에서 쓸 커스텀 GEMM 커널 작성 수단(현재 컴파일 파이프라인과 동일) |
| **Bonsai/PrismML** | g128 삼진=1.71bpw, 배포 밀도 1.95~2.29bpw 기준 |
| 우리 `tinylm/model/ternary.py` | 정본 알고리즘: g128 TWN 임계+L2 alpha, weight기반 AMP-안전 STE |

## 초기 구조 v0 (정확성 우선)
1. **Forward 커널(Triton)**: 입력 활성 `x`[.,I](bf16), 삼진 가중치 `wq`(그룹 alpha 포함) →
   `y = x @ wq^T`. 그룹 스케일 alpha를 커널 내부에서 적용. 우선 **수학적 등가**(현 bf16 경로와 동일 출력) 확보.
2. **Backward**: `grad_x = grad_y @ wq`(dequant wq 재사용), `grad_weight = (grad_y^T @ x) * STE_window(latent w)`.
   STE window는 저장한 latent `w`로 계산. 기존 `_TernarySTE.backward`와 동일 결과여야 함.
3. 타깃: **SM89(4070 Ti Super) + Windows Triton**. 우선 forward만 커널화하고 backward는 기존 STE 유지해도 됨(부분 이득).
4. 검증: 같은 seed에서 loss 곡선이 현 경로와 일치(정확성), step time 단축(속도).

## 고도화 5안 (v0 정상 작동 시)
1. **Dequant+GEMM 융합**: 별도 dequant 텐서 없이 커널 내부에서 삼진 코드→활성 곱(메모리·대역폭↓).
2. **LUT 기반(T-MAC식)**: 삼진×활성 부분곱을 LUT로 사전계산 → CPU 배포 커널과 **단일 구현**으로 통일.
3. **g128 그룹 스케일 1급 시민**: 그룹별 alpha를 커널 레이아웃에 내장(DeepGEMM 원칙), grouped GEMM.
4. **활성 저비트 혼합정밀**: 활성을 int8/FP8(Ada)로, 가중치는 삼진 → 저비트×저비트 GEMM으로 추가 가속.
5. **STE window backward 융합**: backward도 단일 커널로(그라디언트에 window 곱을 커널 내부 처리) → 역전파 오버헤드↓.

## 리스크·대기 사유
- Windows Triton 성숙도, SM89 제약, **학습 backward의 STE 복잡성**(추론 커널엔 없음).
- 우선순위: 아키텍처 검증(P002~P011)·데이터효율(P012)이 앞. **커널은 스케일 단계에서 착수**(이 문서로 준비만).

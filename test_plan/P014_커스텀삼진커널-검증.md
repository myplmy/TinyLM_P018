# 실험계획 P014 — 커스텀 삼진 커널 검증 (정확성 우선 → 속도)  [-done(003)]

> **★2026-08-06 재정의 — P034 단계4(R8)와 같은 물건이 필요하다**
> 결과 014 §11 이 값어치를 수치로 만들었다: `b`(층당) **0.913~1.163 ms**, 그리고
> **분리된 unpack 은 +2.1 ms** 를 얹는다. 따라서 커스텀 커널의 목표는 두 가지가 됐다.
> 1. **융합**(dequant/unpack + GEMM 을 하나로) — 분리 연산의 +2.1 ms 를 없앤다.
> 2. **패킹된 가중치를 직접 읽기** — P034 단계4 가 요구하는 바로 그것.
>
> **즉 P014 와 P034 단계4 는 같은 커널을 필요로 한다.** 따로 하지 말고 한 번에 설계한다.
> ⚠️ **상한은 FLOPs 다.** 커널은 `b` 를 0 으로 못 만든다 — 잘해야 캐시 지역성 개선(§11.3 의 21.5% 급).
> 기대값을 그렇게 잡고 착수한다.

> **★2026-08-06 — 새로운 근거가 생겼다(단, 선결 조건이 붙는다)**
> 종전 판단은 "Triton 커널이 레퍼런스보다 빠르지 않으면 이득 없음" 이었다.
> 결과 016 §10.4 가 다른 각도를 열었다 — batch 1 디코드가 **런치 오버헤드 바운드**로 보인다.
> 그렇다면 이득은 GEMM 을 빠르게 하는 데서가 아니라 **커널 개수를 줄이는 융합**
> (dequant + GEMM 을 하나로)에서 온다. 커스텀 커널이 정확히 할 수 있는 일이다.
> ⚠️ **선결: `run_P030_stage4_layerscaling.bat`.** 층 수를 바꿔도 tok/s 가 안 변하면
> 융합도 소용없다. **가설을 먼저 검정한다.**

## 목적
분리 작성한 커스텀 삼진 커널(`tinylm/model/ternary_kernel.py`, 기본 off)의
(a) **레퍼런스 정확성**(기존 경로와 수학적 동일), (b) **Triton forward 정확성**, (c) **속도**를 검증.

## 원칙
- **기본 학습 무영향**: 플래그 없으면 기존 경로. `--ternary-kernel`(레퍼런스), 추가로
  `--ternary-kernel-triton`(Triton). Triton은 **정확성 통과 전엔 학습에 쓰지 말 것**(오염 위험).

## 단계
### 1) 레퍼런스 정확성 (동일 loss 이어야 함 — 수학적 등가)
커널 레퍼런스 wq = codes*alpha 는 `_TernarySTE` 출력과 **동일**, 이후 F.linear·STE backward도 동일.
따라서 같은 seed면 **step별 loss가 정확히 같아야** 한다(다르면 버그).
```cmd
python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2 --eval-every 20 --tag k_off
python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2 --eval-every 20 --ternary-kernel --tag k_ref
```
→ 두 로그의 loss/val 이 동일한지 확인.

### 2) Triton forward 정확성 (거의 동일 — bf16 누적순서 차 허용)
```cmd
python run100m.py train --arch tied --tiny --data synthetic --tokens 2M --steps 40 --micro-bs 4 --seq 128 --accum 2 --eval-every 20 --ternary-kernel --ternary-kernel-triton --compile --tag k_triton
```
→ k_off/k_ref 대비 loss가 미세오차(±0.01) 내면 통과. 크게 다르면 Triton 커널 버그(→ 자동 폴백 로그 확인·수정).

### 3) 속도 (정확성 통과 후에만)
```cmd
python run100m.py train --arch tied --data ko-en --tokens 50M --steps 200 --micro-bs 8 --seq 1024 --accum 16 --lr 1e-3 --compile --no-ckpt --tag k_spd_off
python run100m.py train --arch tied --data ko-en --tokens 50M --steps 200 --micro-bs 8 --seq 1024 --accum 16 --lr 1e-3 --compile --no-ckpt --ternary-kernel --ternary-kernel-triton --tag k_spd_on
```
→ ms/step 비교.

## 판정
- 1)·2) 통과(loss 동일/미세오차)해야 3) 진행. Triton이 느리거나 부정확하면 **레퍼런스 유지(이득 없음)**
  하고 커널 개선(docs/custom_ternary_kernel.md 고도화 5안).
- 현 커널(int8 codes + bf16 활성, dequant-in-kernel)은 **가중치 대역폭 이득 위주** → 연산 바운드에선
  이득이 작을 수 있음. **FLOP 이득은 고도화 4안(활성 저비트 int8/FP8)에서**.

## 비고
- torch.compile 과 커스텀 Triton 커널은 별개(커널은 compile 그래프 밖 독립 실행). 정확성 1단계는 --compile 없이.
- 커널은 학습 검증용. 배포(추론) 삼진 커널은 T-MAC/llama.cpp 별도 트랙.

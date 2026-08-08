# P027 — CUDA Graphs + torch.compile (CPU 런치 오버헤드 제거)

## 목적
소형 모델은 레이어별 커널이 짧아 **Python/PyTorch 커널 디스패치(CPU) 오버헤드**가 병목이 될 수 있다.
전체 forward-backward 그래프를 CUDA Graphs로 녹화해 CPU 오버헤드를 0에 수렴시켜 GPU 포화율을 높인다.
`torch.compile(mode="reduce-overhead")` 가 이 CUDA Graphs 경로다.

> **전제 주의(정직)**: 우리 학습은 M=micro_bs×seq=8192 대배치라 GEMM 커널이 충분히 커서 **런치 바운드가
> 아닐 수 있다**(연산바운드). "not enough SMs" 는 모델이 작아(N,K 작음) 나는 것이지 CPU 디스패치 문제가
> 아니다. 따라서 이득은 **launch-bound일 때만** 크고, 상한이 낮을 수 있음(method 원장은 10~25% 가능으로 기록).

## 현재 상태 (코드)
- `--compile-mode reduce-overhead` 는 **이미 존재**하나, trainer 주석대로 **임베딩 타잉과 충돌해 backward
  에서 크래시** 가능. grad-accum 경계엔 `cudagraph_mark_step_begin()` 를 각 forward 전에 호출(v6 처리됨).
- 즉 **선결과제 = 타잉-임베딩(emb.weight 를 lm_head 로 재사용) + CUDA Graphs 정적버퍼 충돌 해소.**

## 가설 / 접근
1. **충돌 원인 규명**: CUDA Graphs 는 출력 버퍼를 재사용하는데, factorized emb(`emb_up.weight.t()` →
   `emb.weight`)를 forward·lm_head 양쪽에서 참조 → 버퍼 앨리어싱/mutation 추적 실패 추정.
2. **완화안**: (a) lm_head 를 별도 파라미터로 분리(타잉 해제, 메모리 소폭↑) 후 CUDA Graphs 적용 A/B,
   또는 (b) mark_step_begin 위치·`torch.compiler.cudagraph_mark_step_begin` 호출 시점 조정,
   또는 (c) reduce-overhead 를 **모델 본체에만** 적용하고 lm_head 는 그래프 밖.
3. 안 되면 **default compile 유지 + 부분 CUDA Graphs**(어텐션/MLP 블록만) 실험.

## 실험 매트릭스 (m100·tiny, ko-en, 짧은 steps로 ms/step만)
| 태그 | compile-mode | 확인 |
|---|---|---|
| base | default | 기준 ms/step |
| ro | reduce-overhead | 크래시 여부 → 해소 후 ms/step |
| ro_untied | reduce-overhead + lm_head 분리 | 타잉 해제 시 동작·속도 |

## 판정
- reduce-overhead 가 **크래시 없이** 돌고 default 대비 ms/step **10%+↓** 면 채택(품질 불변 확인).
- launch-bound 아니어서 이득 미미하면(≤5%) 음성결과 기록 후 default 유지.

## 리스크 / 비고
- reduce-overhead 는 **VRAM 약간↑**(그래프 정적버퍼) → 스필벽 주의.
- 동적 분기(anneal·KD skip-forward)와 CUDA Graphs 상성 확인 필요(그래프 재녹화 유발 시 이득 소멸).
- 저비용·저위험(플래그 존재)이라 **P021 벤치와 함께** 빠르게 확인 가능.

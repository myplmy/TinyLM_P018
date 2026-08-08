# 실험계획 P003 — KD + g 스윕 (더 공격적 타잉으로 메모리 추가 감축)  [-done(002)]

## 목적
P002에서 **KD+부모초기화가 g=4 격차를 소멸**(+0.001, 메모리 0)시켰다. 그렇다면
**KD를 얹은 채 타잉을 더 세게(g=8, 깊은 프리셋의 g=6)** 걸어 **메모리를 더 줄이면서도
품질(격차)을 유지**할 수 있는지 확인한다. 성공 시 감축비 1.82→2.08× 이상으로 확대.

## 제약
- `mlp_group(g)` 는 `n_middle` 을 나눠야 함. m100(mid16): g∈{2,4,8}. m100d(mid24): g∈{2,3,4,6,8,12}.
- g가 커질수록 |g| 상승·불안정 가능(P002 t_lora32에서 |g|6.2 관찰) → LR·warmup 감시.
- m100 조건은 **기존 dense 교사 재사용**. m100d 조건은 **m100d dense 교사 신규 학습 필요**.

## 아키텍처·감축(예상)
| 구성 | g | 감축(예상) |
|---|---|---|
| m100 (mid16) | 4 (기준) | 1.82× |
| m100 (mid16) | 8 | ~2.08× |
| m100d (mid24) | 6 | (report로 확인) |
| m100d (mid24) | 8 | (report로 확인) |

## 실행
공통: `--data ko-en --tokens 300M --steps 2289 --micro-bs 8 --seq 1024 --accum 16 --lr 1e-3
--eval-every 100 --compile --init-from --kd` (KD라 --no-ckpt 빼기). g는 `--mlp-group` 로 오버라이드.

```cmd
:: m100 g=8 + KD+init  (dense 교사 재사용)
python run100m.py train --arch tied ... --init-from --kd --mlp-group 8 --tag t_kd_g8

:: (선택) m100d 계열은 먼저 m100d dense 교사부터
python run100m.py train --arch dense --preset m100d ... --no-ckpt          :: m100d dense 교사
python run100m.py train --arch tied  --preset m100d ... --init-from --kd --mlp-group 6 --tag t_kd_m100d_g6
python run100m.py train --arch tied  --preset m100d ... --init-from --kd --mlp-group 8 --tag t_kd_m100d_g8
```

## 판정
- 각 조건 격차 = tied final − 해당 dense final. **KD 적용 후에도 ≤ +0.07 유지되면 그 g 채택** →
  메모리 감축 확대. 격차가 노이즈(±0.03) 넘게 벌어지면 그 g는 과함.
- |g|max 10 이상이면 LR 문제(학습), 아키텍처 문제 아님.

## 산출
- 결과는 `test_result/00N_..._KD-g스윕.md` + 실험목록 갱신. `docs/methods/02_memory.md`(g 스윕 행) 특기 갱신.

## 부속(S5) — warm-start
- g 변형은 처음부터가 아니라 **KD된 g4 모델에서 warm-start**하면 수렴이 빨라진다(속도 전술).

## 확장 — 극단 타잉 t_kd_g16 + 오프라인 KD
- g=16(중간16→MLP 1그룹)도 KD가 격차를 닫는가? `--mlp-group 16 --init-from --kd --tag t_kd_g16`.
  (실측 002: g=8 닫힘 → g=16 한계 탐색.)
- **오프라인 KD 적용 가능**(P015): 동일 dense 교사로 `kdcache` 1회 → `--kd-cache`로 g16 학생을 교사 forward 없이
  빠르게. 단 top-k 근사라 온라인 KD와 미세차 → 먼저 tiny 검증 후 사용.
  (결과 004에서 **오프라인 top-k KD 는 부적합** 판정 — 고엔트로피 교사에서 질량 손실.)

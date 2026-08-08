# 실험계획 P004 — Kimi Delta Attention(KDA) 적용 (롱컨텍스트 KV 절감)

## 목적
KV 캐시를 대폭 줄여(롱컨텍스트) 배포 메모리·컨텍스트 한계를 개선. KDA(채널게이팅 선형 어텐션)를
softmax 어텐션과 3:1 하이브리드로 섞으면 KV ~75%↓, 긴 문맥 디코딩 배속(참고: RULER 128k 우수).

## 선결 과제 (구현 필요)
- **KDA 레이어 구현**: 선형 어텐션 chunkwise 커널 필요(예: `fla`/flash-linear-attention 라이브러리
  또는 커스텀). **GPU 전용**(CPU 커널 없음) — 배포 타깃이 CPU면 이 이득은 GPU/모바일GPU 한정.
- **NoPE**: KDA 레이어는 RoPE 대신 NoPE가 유리(참고). softmax(CLA) 레이어는 RoPE 유지.
- **아키텍처를 config로 선택 가능하게**(P005/(5) 참조): `attn_kind ∈ {softmax_cla, kda}` 를 층별로
  지정 → CLA-only vs KDA-hybrid 를 **한 코드베이스에서 프리셋만 바꿔 비교**.

## 아키텍처(안)
- 중간 16층을 KDA:softmax = 3:1 로: 12층 KDA(NoPE) + 4층 softmax(CLA2, RoPE).
- prelude/coda는 softmax 유지(경계 조정).

## 실행(구현 후)
```cmd
:: CLA-only 기준선(현재) vs KDA-hybrid 를 동일 토큰으로 비교
python run100m.py train --arch tied --preset m100     ... --tag t_cla     :: 기준(softmax+CLA2)
python run100m.py train --arch tied --preset m100_kda ... --tag t_kda     :: KDA 3:1 하이브리드
python run100m.py compare --tag t_cla --vs t_kda
```

## 판정
- **KV 캐시 KB/token**(report) 감소 확인(목표 ~4배↓) + **품질 격차**(val)가 허용범위(≤+0.07)인지.
- 롱컨텍스트(예: seq 2048/4096)에서 메모리·속도 이점 측정.

## 리스크·비고
- 구현 난이도 높음(선형 어텐션 커널). CPU 배포엔 직접 기여 X(GPU/모바일GPU용).
- SISA와 배타적. 먼저 (5) config 기반 아키텍처 선택 구조를 갖춘 뒤 진행 권장.

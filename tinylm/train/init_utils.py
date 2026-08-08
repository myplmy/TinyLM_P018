"""부모(dense) 체크포인트 활용 — KD 교사 로드 & 부모초기화(RRT식 average-init)."""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import TMTConfig
from ..model import TiedMLPTransformer


def _strip(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def _cfg_of(model):
    """모델의 cfg. `load_dense` 가 돌려주는 것과 학생 둘 다 `.cfg` 를 갖는다."""
    return model.cfg


@torch.no_grad()
def _svd_emb_init(student, teacher) -> bool:
    """★P046 단계2(2026-08-07) — **E 가 줄어든 임베딩을 난수가 아니라 SVD 절단으로 시작한다.**

    ## 왜 필요한가 — 결과 030 의 판정이 이것 때문에 무효였다

    `--emb-rank 128` 런(`mC_e128`)은 임베딩만 **난수**로 시작했다. 그 결과:
      · step 0 의 `ce = 10.4051` ≈ `ln(32768) = 10.397` = **완전 무작위 출력**
        (같은 밤의 `mC_g16` 은 7.7742 로 부모초기화 이득이 살아 있었다)
      · **`grad_max = 19.32`** vs 기준선 0.847 — `CLAUDE.md` 판정 기준의
        *"> +0.15 는 grad_max 먼저 확인, **10 이상이면 학습 문제**"* 에 정확히 걸린다
    → **paired Δ +0.1768 은 "E=128 이 나쁘다" 가 아니라 "초기화가 깨졌다" 였다.**

    ## 무엇을 하는가 — 절단이 아니라 **최적 저랭크 근사**

    factorized embedding 의 실효 가중치는 두 행렬의 곱이다:
        `W = emb(V × E) @ emb_up(E × dim)`        shape (V, dim), 랭크 ≤ E
    학생의 E' < E 가 표현할 수 있는 최선은 **`W` 의 랭크 E' 최적 근사**이고,
    그건 Eckart–Young 정리로 **`W` 의 상위 E' 특이값**이다:
        `W ≈ U_k S_k V_k^T` → `emb' = U_k √S_k`, `emb_up' = √S_k V_k^T`
    `√S` 로 나눠 갖는 것은 두 인자의 스케일을 맞춰 초기 gradient 를 고르게 하려는 것이다.

    ⚠️ **이것이 "E=128 을 공짜로 만든다" 는 뜻은 아니다.** 로짓 랭크가 E' 로 제한되는
    구조적 한계는 그대로다. 이 함수는 **그 한계만 남기고 초기화 핸디캡을 걷어낸다** —
    즉 P046 을 "판정 불가" 에서 "판정 가능" 으로 옮기는 것이 전부다.

    ⚠️ **헤드가 임베딩과 tie** 돼 있으므로(결과 025) 이 근사는 입력과 출력에 **동시에**
    영향을 준다. 그래서 `W` 를 근사하는 것이 옳다 — `emb` 만 절단하면 출력 쪽이 깨진다.

    반환: 이식했으면 True, 조건이 안 맞으면 False(호출부가 난수 경로로 간다).
    """
    se, te = student.emb.weight, teacher.emb.weight
    if student.emb_up is None or teacher.emb_up is None:
        return False
    if se.shape[0] != te.shape[0]:              # 어휘가 다르면 이 방법이 성립하지 않는다
        return False
    k, k_t = se.shape[1], te.shape[1]
    if k >= k_t:                                # E 를 **키운** 경우는 절단이 아니다 → 난수
        return False

    W = (te.float() @ teacher.emb_up.weight.float())          # (V, dim), 랭크 ≤ k_t
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = torch.sqrt(S[:k].clamp_min(0))
    student.emb.weight.data.copy_((U[:, :k] * r).to(se.dtype))
    student.emb_up.weight.data.copy_((r[:, None] * Vh[:k]).to(student.emb_up.weight.dtype))

    kept = float((S[:k] ** 2).sum() / (S ** 2).sum().clamp_min(1e-12))
    print(f"[init] ★임베딩 SVD 절단 이식 — E {k_t} -> {k}  (난수 아님, P046 단계2)")
    print(f"[init]   보존된 스펙트럼 에너지 {kept*100:.2f}%  "
          f"(σ_1 {float(S[0]):.3f} / σ_{k} {float(S[k-1]):.3f} / σ_{k_t} {float(S[-1]):.3f})")
    print(f"[init] ⚠️ 로짓 랭크가 {k} 로 제한되는 **구조적** 한계는 그대로다 — "
          f"이 이식은 **초기화 핸디캡만** 걷어낸다.")
    return True


def load_dense(path, device):
    """dense 체크포인트를 그 cfg로 복원한다(KD 교사 또는 초기화 소스)."""
    st = torch.load(path, map_location=device)
    cfg = TMTConfig(**st["cfg"])
    m = TiedMLPTransformer(cfg).to(device)
    m.load_state_dict(_strip(st["model"]))
    return m, cfg


@torch.no_grad()
def init_from_dense(student: TiedMLPTransformer, dense_path, device):
    """dense 가중치를 tied 학생에 이식.
      - 임베딩·최종노름·어텐션(q/o, 소유층 k/v)·adaLN·gate: 인덱스 그대로 복사
      - prelude/coda MLP: 그대로 복사
      - 중간 MLP: dense의 mlp_group개 층을 평균 내어 공유 인스턴스에 넣음(RRT average-init)
    LoRA(있으면)는 U=0 초기화라 시작 시 항등 → 별도 처리 불필요."""
    teacher, _ = load_dense(dense_path, device)
    g = student.cfg.mlp_group

    # ★2026-08-07(P046) — `--emb-rank` 로 E 를 바꾸면 임베딩 shape 이 교사와 다르다.
    #   종전에는 `load_state_dict` 가 **strict 라 즉사**했다. 지금은 **건너뛰고 크게 알린다** —
    #   조용히 넘기면 "부모초기화했다" 고 믿는 채로 임베딩만 난수인 런이 생긴다.
    #   어텐션·MLP shape 은 E 에 의존하지 않으므로 그쪽 이식은 그대로 유효하다.
    if student.emb.weight.shape == teacher.emb.weight.shape:
        student.emb.load_state_dict(teacher.emb.state_dict())
        if student.emb_up is not None:
            student.emb_up.load_state_dict(teacher.emb_up.state_dict())
    elif _svd_emb_init(student, teacher):
        pass                                   # ★아래 함수가 알린다
    else:
        print(f"[init] ★★경고: 임베딩 shape 불일치 — 학생 {tuple(student.emb.weight.shape)} vs "
              f"교사 {tuple(teacher.emb.weight.shape)}")
        print(f"[init] ★임베딩·emb_up 은 **부모초기화하지 않고 난수로 시작**한다"
              f"(어텐션·MLP 는 정상 이식). `--emb-rank` 를 바꾼 런이면 의도된 동작이다.")
        print(f"[init] ⚠️ 이 런을 E 가 같은 런과 비교할 때 **초기화 조건이 다르다**는 것을 명시할 것.")
    student.norm_f_scale.data.copy_(teacher.norm_f_scale.data)

    # ★2026-08-07(P048) — **깊이·prelude/coda 개수가 다르면 `zip` 이 조용히 자른다.**
    #   `m100R1p`(1+16+1 = 18층)를 `m100` dense(2+16+2 = 20층)에서 초기화하면
    #   학생 층 0..17 이 교사 층 0..17 을 받는데, **학생의 층 17 은 coda 인데 교사의
    #   층 17 은 중간층**이다. 값은 들어가지만 **역할이 다른 층의 가중치**다.
    #   `pre_mlps`·`coda_mlps` 도 학생 개수만큼만 돌아 교사의 뒷것을 버린다.
    #   ⚠️ 이건 버그가 아니라 **의도된 부분 이식**이다. 다만 조용하면 안 된다 —
    #   "부모초기화했다" 고 믿는 채로 조건이 다른 런이 생기고, 그게 결과 015·026 계열
    #   사고의 형태다(계상되지 않은 차이). 그래서 **크게 알리고 무엇이 어긋났는지 적는다**.
    ns, nt = len(student.layers), len(teacher.layers)
    if ns != nt or student.cfg.n_prelude != _cfg_of(teacher).n_prelude \
            or student.cfg.n_coda != _cfg_of(teacher).n_coda:
        tc = _cfg_of(teacher)
        print(f"[init] ★★경고: 구조 불일치 — 학생 층 {ns}(prelude {student.cfg.n_prelude}/"
              f"coda {student.cfg.n_coda}) vs 교사 층 {nt}(prelude {tc.n_prelude}/coda {tc.n_coda})")
        print(f"[init] ★앞 {min(ns, nt)}개 층만 **인덱스 순서로** 이식한다 — "
              f"학생의 coda 층이 교사의 중간층에서 값을 받을 수 있다(역할 불일치).")
        print(f"[init] ★prelude/coda MLP 는 학생 개수만큼만 받고 교사의 나머지는 버린다.")
        print(f"[init] ⚠️ 이 런을 층수가 같은 런과 비교할 때 **초기화 조건이 다르다**는 것을 명시할 것.")

    for ls, lt in zip(student.layers, teacher.layers):
        ls.attn.q_proj.weight.data.copy_(lt.attn.q_proj.weight.data)
        ls.attn.o_proj.weight.data.copy_(lt.attn.o_proj.weight.data)
        if ls.attn.owns_kv:                     # 교사(dense, cla1)는 모든 층이 k/v 보유
            ls.attn.k_proj.weight.data.copy_(lt.attn.k_proj.weight.data)
            ls.attn.v_proj.weight.data.copy_(lt.attn.v_proj.weight.data)
        for a in ("a_scale", "a_shift", "m_scale", "m_shift", "gates"):
            getattr(ls, a).data.copy_(getattr(lt, a).data)

    for j, mlp_s in enumerate(student.pre_mlps):
        mlp_s.load_state_dict(teacher.pre_mlps[j].state_dict())
    for j, mlp_s in enumerate(student.coda_mlps):
        mlp_s.load_state_dict(teacher.coda_mlps[j].state_dict())
    for j, mlp_s in enumerate(student.mid_mlps):     # 그룹 평균
        members = [teacher.mid_mlps[j * g + k] for k in range(g)]
        ref = dict(mlp_s.named_parameters())
        for name, ps in ref.items():
            avg = sum(dict(mm.named_parameters())[name].data for mm in members) / g
            ps.data.copy_(avg)
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[init] dense 부모초기화 완료 (중간 MLP는 {g}층 평균)  <- {Path(dense_path).name}")

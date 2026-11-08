"""
SPSA-аналоги: 7 алгоритмов распределённой оптимизации нулевого порядка.

Все функции имеют единый интерфейс:
    f(controllers, step, alpha, beta, gamma) -> None
и обновляют ctrl.theta_time / ctrl.theta_wait на месте.

Источники:
    aspsa   — Anonymous et al., IEEE TAC 2026
    kw      — Sahu, Jakovetic, Bajovic, Kar
    zo_pgd  — Akhavan, Pontil, Tsybakov, NeurIPS 2021
    sp_gt   — Mhanna & Assaad, ICML 2023
    zo_gt   — Cheng, Yu, Fan, Xiao, IFAC 2023
    pd_2pt  — Yi, Li, Yang, Xie, Chai, Johansson, IEEE TAC 2021
    pd_1pt  — Yi et al., IEEE TAC 2021
"""

import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# =============================================================================
# Персистентное состояние между вызовами (моментум, трекеры градиентов)
# =============================================================================
_STATE: Dict[str, Dict[int, Any]] = {}

# =============================================================================
# Per-step parameter trace (collected when trace is enabled).
# Each entry: {step, variant, alpha_k/alpha_t, kappa_k, pert, grad_norm_mean}
# =============================================================================
_TRACE_LOG: List[Dict[str, Any]] = []
_TRACE_ENABLED: bool = False


def enable_trace(flag: bool = True) -> None:
    global _TRACE_ENABLED
    _TRACE_ENABLED = flag


def clear_trace() -> None:
    _TRACE_LOG.clear()


def get_trace() -> List[Dict[str, Any]]:
    return list(_TRACE_LOG)


def _st(algo: str) -> Dict[int, Any]:
    if algo not in _STATE:
        _STATE[algo] = {}
    return _STATE[algo]


def reset_state() -> None:
    """Вызывать перед каждым новым запуском симуляции."""
    _STATE.clear()


# =============================================================================
# Расписания шага (Robbins-Monro, Spall 1998)
# Параметры: s=0.602, t=0.101 — стандарт SPSA.
# A=10 — смягчение на старте; a, b — масштабы, настраиваемые снаружи.
# =============================================================================

def _decay_alpha(step: int, a: float, A: float = 10.0, s: float = 0.602) -> float:
    """α_t = a / (t + 1 + A)^s  — убывающий шаг градиентного спуска."""
    return a / (step + 1 + A) ** s


def _decay_pert(step: int, b: float, t: float = 0.101) -> float:
    """β_t = b / (t + 1)^t  — убывающий масштаб возмущения."""
    return b / (step + 1) ** t


# =============================================================================
# Вспомогательные функции (зеркало main.py, без импорта из него)
# =============================================================================

def task_feature_vector(task_type, features, dimension, num_types=5):
    """One coordinate per category plus an optional independent semantic feature."""
    vector = np.zeros(dimension)
    vector[task_type] = 1.0
    if dimension > num_types:
        vector[num_types] = float(np.mean(features))
    return vector


def _proc_loss(theta: np.ndarray, obs) -> float:
    """Raw MSE for time prediction."""
    if not obs:
        return 0.0
    n = len(theta)
    errs = []
    for o in list(obs)[-25:]:
        x = task_feature_vector(o.task_type, o.features, n, getattr(o, "num_types", 5))
        errs.append((float(np.dot(theta, x)) - o.true_time) ** 2)
    return float(np.mean(errs)) if errs else 0.0


def _wait_loss(theta: np.ndarray, obs, delta: float = 0.1) -> float:
    """One-sided hinge: penalize underprediction only (conservative upper bound).
    Semantically correct for routing: overestimating wait is safer than underestimating."""
    if not obs:
        return 0.0
    n = len(theta)
    errs = []
    for o in list(obs)[-25:]:
        x = task_feature_vector(o.task_type, o.features, n, getattr(o, "num_types", 5))
        slack = float(np.dot(theta, x)) - o.true_wait - delta
        errs.append(max(0.0, -slack) ** 2)
    return float(np.mean(errs)) if errs else 0.0


def _quality_loss(theta: np.ndarray, obs) -> float:
    """Rolling response-quality MSE used by A-SPSA."""
    if not obs:
        return 0.0
    errs = []
    for features, true_quality in list(obs)[-25:]:
        pred = float(np.clip(np.dot(theta, features), 0.0, 1.0))
        errs.append((pred - float(true_quality)) ** 2)
    return float(np.mean(errs)) if errs else 0.0


def _comm_graph(step: int, n: int) -> np.ndarray:
    """Случайный временной граф (Poisson-степень, макс. 2 соседа)."""
    B = np.zeros((n, n))
    if n <= 1:
        return B
    for i in range(n):
        deg = min(np.random.poisson(1.5) + 1, n - 1)
        cands = [x for x in range(n) if x != i]
        for j in np.random.choice(cands, min(deg, 2), replace=False):
            B[i, j] = 0.5
            B[j, i] = 0.5
    return B


def _row_stochastic(B: np.ndarray) -> np.ndarray:
    """Преобразует матрицу смежности в строчно-стохастическую W."""
    W = B.copy()
    for i in range(W.shape[0]):
        rs = W[i].sum()
        if rs <= 1.0:
            W[i, i] = 1.0 - rs   # добавляем self-weight
        else:
            W[i] /= rs            # нормализуем строку (off-diagonal уже > 1)
    return W


def _consensus(B: np.ndarray, j: int, controllers, attr: str) -> np.ndarray:
    nbrs = np.where(B[j] > 0)[0]
    if len(nbrs) == 0:
        return np.zeros_like(getattr(controllers[j], attr))
    return sum(
        B[j, jp] * (getattr(controllers[j], attr) - getattr(controllers[jp], attr))
        for jp in nbrs
    )


def _clip_t(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.3, 15.0)


def _clip_w(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.1, 8.0)


# =============================================================================
# Ограничение для Primal-Dual методов
# h_j(θ) = mean(θ_time) − C_budget  (мягкое ограничение на очередь)
# =============================================================================
_C_BUDGET = 3.0


def _h(theta_time: np.ndarray, num_types: int = 5) -> float:
    return float(np.mean(theta_time[:num_types])) - _C_BUDGET


def _grad_h(theta_time: np.ndarray, num_types: int = 5) -> np.ndarray:
    grad = np.zeros_like(theta_time)
    count = min(len(theta_time), num_types)
    grad[:count] = 1.0 / count
    return grad


# =============================================================================
# 1. A-SPSA — Algorithm (12) из Anonymous et al., IEEE TAC 2026
#    Трёхточечная схема с квадратичным суррогатом:
#      x̃_k  — точка зондирования (probing point)
#      θ̄_k  — оценка после шага спуска
#      z̄_k  — центр суррогата (memory anchor)
#    Адаптивный α_k из ур. (10): α²/h = (1-α)γ_k + α(µ-η)
#    Обновление кривизны γ_k из ур. (11): γ_k = (1-α_{k-1})γ_{k-1} + α_{k-1}(µ-η)
# =============================================================================

def _solve_alpha(h: float, kappa: float, mu_eta: float) -> float:
    """Eq (10): α²/h = (1-α)γ + α(µ-η)  →  α² + α·h·(γ-(µ-η)) - h·γ = 0."""
    b = h * (kappa - mu_eta)
    disc = max(0.0, b * b + 4.0 * h * kappa)
    return float(np.clip((-b + np.sqrt(disc)) / 2.0, 0.01, 0.99))


def _aspsa_step(
    S: dict, tag: str, controllers, step: int,
    h: float, beta: float, gamma: float,
    mu: float, eta: float,
    fixed_alpha: float | None = None,
    alpha_cap: float = 1.0,
) -> None:
    """Shared inner loop for all A-SPSA variants."""
    B    = _comm_graph(step, len(controllers))
    pert = _decay_pert(step, beta)
    mu_eta = mu - eta

    # --- shared scalar state (γ_k, α_{k-1}) ---
    if "__alg__" not in S:
        S["__alg__"] = {"kappa": mu + h, "prev_alpha": 0.5}
    alg        = S["__alg__"]
    kappa_prev = alg["kappa"]
    prev_alpha = alg["prev_alpha"]

    # Eq (11): γ_k = (1-α_{k-1})γ_{k-1} + α_{k-1}(µ-η)
    kappa_k = max((1.0 - prev_alpha) * kappa_prev + prev_alpha * mu_eta, 1e-8)

    # Eq (10): solve for α_k  (or use fixed value for ablations); cap at alpha_cap
    if fixed_alpha is not None:
        alpha_k = fixed_alpha
    else:
        alpha_k = min(_solve_alpha(h, kappa_k, mu_eta), alpha_cap)

    # r = α_k · γ_{k-1} / γ_k  (weight for surrogate center in probing point)
    r = alpha_k * kappa_prev / kappa_k

    new_t, new_w, new_q = [], [], []
    for j, ctrl in enumerate(controllers):
        cid = ctrl.id
        dim = len(ctrl.theta_time)

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            new_q.append(ctrl.theta_quality.copy())
            continue

        if cid not in S:
            S[cid] = {
                "theta_bar_t": ctrl.theta_time.copy(),
                "theta_bar_w": ctrl.theta_wait.copy(),
                "z_bar_t":     np.zeros(dim),
                "z_bar_w":     np.zeros(dim),
                "theta_bar_q": ctrl.theta_quality.copy(),
                "z_bar_q":     np.zeros(len(ctrl.theta_quality)),
            }

        S[cid].setdefault("theta_bar_q", ctrl.theta_quality.copy())
        S[cid].setdefault("z_bar_q", np.zeros(len(ctrl.theta_quality)))

        tb_t = S[cid]["theta_bar_t"]
        tb_w = S[cid]["theta_bar_w"]
        zb_t = S[cid]["z_bar_t"]
        zb_w = S[cid]["z_bar_w"]
        tb_q = S[cid]["theta_bar_q"]
        zb_q = S[cid]["z_bar_q"]

        # Probing point per paper Eq (12):
        # x̃_k = (α_k·γ_{k-1}·z̄ + γ_k·θ̄) / (γ_{k-1} + α_k·(μ-η))
        denom = kappa_prev + alpha_k * mu_eta   # γ_{k-1} + α_k·(μ-η)
        xt_t  = (alpha_k * kappa_prev * zb_t + kappa_k * tb_t) / denom
        xt_w  = (alpha_k * kappa_prev * zb_w + kappa_k * tb_w) / denom
        xt_q  = (alpha_k * kappa_prev * zb_q + kappa_k * tb_q) / denom

        # SPSA gradient estimate at x̃ — shared perturbation direction for F1 and F2
        d = 2 * np.random.randint(0, 2, dim) - 1   # one Rademacher draw, used for both
        g_t = (_proc_loss(xt_t + pert * d, ctrl.obs_history) -
               _proc_loss(xt_t - pert * d, ctrl.obs_history)) / (2 * pert * d)
        g_w = (_wait_loss(xt_w + pert * d, ctrl.obs_history) -
               _wait_loss(xt_w - pert * d, ctrl.obs_history)) / (2 * pert * d)
        d_q = 2 * np.random.randint(0, 2, len(ctrl.theta_quality)) - 1
        quality_obs = getattr(ctrl, "quality_obs_history", ())
        g_q = (_quality_loss(xt_q + pert * d_q, quality_obs) -
               _quality_loss(xt_q - pert * d_q, quality_obs)) / (2 * pert * d_q)

        # Consensus at probing point
        nbrs = np.where(B[j] > 0)[0]
        if len(nbrs):
            g_t += gamma * sum(B[j, jp] * (xt_t - controllers[jp].theta_time) for jp in nbrs)
            g_w += gamma * sum(B[j, jp] * (xt_w - controllers[jp].theta_wait) for jp in nbrs)
            g_q += gamma * sum(B[j, jp] * (xt_q - controllers[jp].theta_quality) for jp in nbrs)

        # θ̄_{k} = x̃_k − h · ḡ_k
        new_tb_t = xt_t - h * g_t
        new_tb_w = xt_w - h * g_w
        new_tb_q = xt_q - h * g_q

        # z̄_k = (1/γ_k)·[(1-α_k)·γ_{k-1}·z̄_{k-1} + α_k·(µ-η)·x̃_k − α_k·ḡ_k]
        coeff = 1.0 / kappa_k
        new_zb_t = coeff * ((1-alpha_k)*kappa_prev*zb_t + alpha_k*mu_eta*xt_t - alpha_k*g_t)
        new_zb_w = coeff * ((1-alpha_k)*kappa_prev*zb_w + alpha_k*mu_eta*xt_w - alpha_k*g_w)
        new_zb_q = coeff * ((1-alpha_k)*kappa_prev*zb_q + alpha_k*mu_eta*xt_q - alpha_k*g_q)

        S[cid]["theta_bar_t"] = new_tb_t
        S[cid]["theta_bar_w"] = new_tb_w
        S[cid]["z_bar_t"]     = new_zb_t
        S[cid]["z_bar_w"]     = new_zb_w
        S[cid]["theta_bar_q"] = new_tb_q
        S[cid]["z_bar_q"]     = new_zb_q

        new_t.append(new_tb_t)
        new_w.append(new_tb_w)
        new_q.append(new_tb_q)

    alg["kappa"]      = kappa_k
    alg["prev_alpha"] = alpha_k

    logger.debug(
        "[A-SPSA tag=%s] step=%d  alpha_k=%.5f  kappa_k=%.5f  pert=%.5f",
        tag, step, alpha_k, kappa_k, pert,
    )
    if _TRACE_ENABLED:
        _TRACE_LOG.append({
            "step": step, "variant": tag,
            "alpha_k": float(alpha_k), "kappa_k": float(kappa_k), "pert": float(pert),
        })

    for ctrl, t, w, q in zip(controllers, new_t, new_w, new_q):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)
        ctrl.theta_quality = np.clip(q, -2.0, 2.0)


def aspsa_consensus(
    controllers, step: int,
    alpha: float = 0.05, beta: float = 0.1, gamma: float = 0.02,
    beta_nes_max: float = 1.0,
) -> None:
    """
    A-SPSA Algorithm (12) — Anonymous et al., IEEE TAC 2026.
    alpha → h (размер шага / обратная константа Липшица).
    beta_nes_max → cap on adaptive step alpha_k (implements beta_max from the paper).
    """
    h  = alpha
    mu = 0.5 * h
    eta = 0.1 * h
    _aspsa_step(_st("aspsa"), "aspsa", controllers, step, h, beta, gamma, mu, eta,
                alpha_cap=beta_nes_max)


# =============================================================================
# 2. Distributed Kiefer-Wolfowitz
#    Sahu, Jakovetic, Bajovic, Kar
# =============================================================================

def kw_consensus(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
) -> None:
    """
    Распределённый метод Кифера-Вольфовица: конечные разности по каждой
    координате. Точная покоординатная оценка градиента — 2·dim вызовов.
    Сходимость O(1/√k), совпадает с централизованным аналогом.
    """
    B = _comm_graph(step, len(controllers))
    alpha_t = _decay_alpha(step, alpha)
    pert    = _decay_pert(step, beta)
    logger.debug("[KW] step=%d  alpha_t=%.5f  pert=%.5f", step, alpha_t, pert)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "kw", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(pert)})
    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        dim = len(ctrl.theta_time)

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        g_t = np.zeros(dim)
        g_w = np.zeros(dim)

        for k in range(dim):
            e = np.zeros(dim); e[k] = 1.0
            g_t[k] = (_proc_loss(ctrl.theta_time + pert * e, ctrl.obs_history) -
                      _proc_loss(ctrl.theta_time - pert * e, ctrl.obs_history)) / (2 * pert)
            g_w[k] = (_wait_loss(ctrl.theta_wait + pert * e, ctrl.obs_history) -
                      _wait_loss(ctrl.theta_wait - pert * e, ctrl.obs_history)) / (2 * pert)

        c_t = _consensus(B, j, controllers, "theta_time")
        c_w = _consensus(B, j, controllers, "theta_wait")

        new_t.append(ctrl.theta_time - alpha_t * (g_t + gamma * c_t))
        new_w.append(ctrl.theta_wait - alpha_t * (g_w + gamma * c_w))

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)


# =============================================================================
# 3. Zero-Order Projected GD — сферическое сглаживание
#    Akhavan, Pontil, Tsybakov, NeurIPS 2021
# =============================================================================

def zo_pgd_consensus(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
) -> None:
    """
    Распределённый ZO projected gradient descent со сферическим сглаживанием.

    Оценщик: (d/2σ)(f(θ+σu) − f(θ−σu))·u,  u ~ Uniform(S^{d-1}).
    Устойчив к adversarial шуму (несимметричному, скоррелированному).
    Проекция реализована через clipping на допустимое множество.
    """
    B = _comm_graph(step, len(controllers))
    alpha_t = _decay_alpha(step, alpha)
    sigma   = _decay_pert(step, beta)
    logger.debug("[ZO-PGD] step=%d  alpha_t=%.5f  sigma=%.5f", step, alpha_t, sigma)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "zo_pgd", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(sigma)})
    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        dim = len(ctrl.theta_time)

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        u_t = np.random.randn(dim); u_t /= np.linalg.norm(u_t) + 1e-8
        u_w = np.random.randn(dim); u_w /= np.linalg.norm(u_w) + 1e-8

        g_t = (dim / (2 * sigma)) * (
            _proc_loss(ctrl.theta_time + sigma * u_t, ctrl.obs_history) -
            _proc_loss(ctrl.theta_time - sigma * u_t, ctrl.obs_history)
        ) * u_t

        g_w = (dim / (2 * sigma)) * (
            _wait_loss(ctrl.theta_wait + sigma * u_w, ctrl.obs_history) -
            _wait_loss(ctrl.theta_wait - sigma * u_w, ctrl.obs_history)
        ) * u_w

        c_t = _consensus(B, j, controllers, "theta_time")
        c_w = _consensus(B, j, controllers, "theta_wait")

        new_t.append(ctrl.theta_time - alpha_t * (g_t + gamma * c_t))
        new_w.append(ctrl.theta_wait - alpha_t * (g_w + gamma * c_w))

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)


# =============================================================================
# 4. Single-Point Gradient Tracking
#    Mhanna & Assaad, ICML 2023
# =============================================================================

def sp_gt_consensus(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
) -> None:
    """
    Одноточечный distributed gradient tracking.

    Один вызов функции на контроллер на шаг.
    Трекер y корректирует смещение одноточечного оценщика.

    Обновления:
        y_i ← Σ_j W_ij·y_j + (g_new − g_old)   ← gradient tracking
        x_i ← Σ_j W_ij·x_j − α·y_i              ← primal
    Сходимость: O(1/K^{1/3}), non-convex.
    """
    S = _st("spgt")
    B = _comm_graph(step, len(controllers))
    W = _row_stochastic(B)
    alpha_t = _decay_alpha(step, alpha)
    sigma   = _decay_pert(step, beta)
    logger.debug("[SP-GT] step=%d  alpha_t=%.5f  sigma=%.5f", step, alpha_t, sigma)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "sp_gt", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(sigma)})

    # Текущие значения y для averaging — фиксируем до обновления
    y_t_cur = {ctrl.id: S.get(ctrl.id, {}).get("y_t", np.zeros_like(ctrl.theta_time))
               for ctrl in controllers}
    y_w_cur = {ctrl.id: S.get(ctrl.id, {}).get("y_w", np.zeros_like(ctrl.theta_wait))
               for ctrl in controllers}

    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        cid = ctrl.id
        dim = len(ctrl.theta_time)

        if cid not in S:
            S[cid] = {
                "y_t":  np.zeros(dim),
                "y_w":  np.zeros(dim),
                "pg_t": np.zeros(dim),
                "pg_w": np.zeros(dim),
            }

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        u_t = np.random.randn(dim); u_t /= np.linalg.norm(u_t) + 1e-8
        u_w = np.random.randn(dim); u_w /= np.linalg.norm(u_w) + 1e-8

        # Одноточечный сферический оценщик: (d/σ)·f(θ+σu)·u  (Mhanna & Assaad ICML 2023)
        # Наша статья пишет d/σ² — вероятно опечатка: d/σ² даёт растущий шаг и расходится.
        g_t = (dim / sigma) * _proc_loss(ctrl.theta_time + sigma * u_t, ctrl.obs_history) * u_t
        g_w = (dim / sigma) * _wait_loss(ctrl.theta_wait + sigma * u_w, ctrl.obs_history) * u_w

        nbrs = np.where(B[j] > 0)[0]

        # Gradient tracking: y ← W·y + (g_new − g_old)
        y_t_new = W[j, j] * y_t_cur[cid] + (g_t - S[cid]["pg_t"])
        y_w_new = W[j, j] * y_w_cur[cid] + (g_w - S[cid]["pg_w"])
        for jp in nbrs:
            y_t_new += W[j, jp] * y_t_cur[controllers[jp].id]
            y_w_new += W[j, jp] * y_w_cur[controllers[jp].id]

        # Primal: x ← W·x − α·y
        nt = W[j, j] * ctrl.theta_time - alpha_t * y_t_new
        nw = W[j, j] * ctrl.theta_wait - alpha_t * y_w_new
        for jp in nbrs:
            nt += W[j, jp] * controllers[jp].theta_time
            nw += W[j, jp] * controllers[jp].theta_wait

        S[cid]["pg_t"] = g_t.copy()
        S[cid]["pg_w"] = g_w.copy()
        S[cid]["y_t"]  = y_t_new
        S[cid]["y_w"]  = y_w_new

        new_t.append(nt)
        new_w.append(nw)

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)


# =============================================================================
# 5. Zeroth-Order Gradient Tracking + проекция
#    Cheng, Yu, Fan, Xiao, IFAC 2023
# =============================================================================

def zo_gt_consensus(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
) -> None:
    """
    ZO gradient tracking с проекцией на неодинаковые допустимые множества X_i.

    Двухточечный SPSA-оценщик + gradient tracking + Π_{X_i}.
    Каждый контроллер имеет слегка смещённые границы — неодинаковые X_i.
    Сходимость: O(ln T / √T).

    Обновления:
        y_i ← Σ_j W_ij·y_j + (g_new − g_old)
        x_i ← Π_{X_i}[Σ_j W_ij·x_j − α·y_i]
    """
    S = _st("zogt")
    B = _comm_graph(step, len(controllers))
    W = _row_stochastic(B)
    alpha_t = _decay_alpha(step, alpha)
    pert    = _decay_pert(step, beta)
    logger.debug("[ZO-GT] step=%d  alpha_t=%.5f  pert=%.5f", step, alpha_t, pert)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "zo_gt", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(pert)})

    y_t_cur = {ctrl.id: S.get(ctrl.id, {}).get("y_t", np.zeros_like(ctrl.theta_time))
               for ctrl in controllers}
    y_w_cur = {ctrl.id: S.get(ctrl.id, {}).get("y_w", np.zeros_like(ctrl.theta_wait))
               for ctrl in controllers}

    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        cid = ctrl.id
        dim = len(ctrl.theta_time)

        if cid not in S:
            S[cid] = {
                "y_t":  np.zeros(dim),
                "y_w":  np.zeros(dim),
                "pg_t": np.zeros(dim),
                "pg_w": np.zeros(dim),
            }

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        # Двухточечный SPSA-псевдоградиент
        d_t = 2 * np.random.randint(0, 2, dim) - 1
        d_w = 2 * np.random.randint(0, 2, dim) - 1

        g_t = (_proc_loss(ctrl.theta_time + pert * d_t, ctrl.obs_history) -
               _proc_loss(ctrl.theta_time - pert * d_t, ctrl.obs_history)) / (2 * pert * d_t)
        g_w = (_wait_loss(ctrl.theta_wait + pert * d_w, ctrl.obs_history) -
               _wait_loss(ctrl.theta_wait - pert * d_w, ctrl.obs_history)) / (2 * pert * d_w)

        nbrs = np.where(B[j] > 0)[0]

        # Gradient tracking
        y_t_new = W[j, j] * y_t_cur[cid] + (g_t - S[cid]["pg_t"])
        y_w_new = W[j, j] * y_w_cur[cid] + (g_w - S[cid]["pg_w"])
        for jp in nbrs:
            y_t_new += W[j, jp] * y_t_cur[controllers[jp].id]
            y_w_new += W[j, jp] * y_w_cur[controllers[jp].id]

        # Primal step с consensus
        pre_t = W[j, j] * ctrl.theta_time - alpha_t * y_t_new
        pre_w = W[j, j] * ctrl.theta_wait - alpha_t * y_w_new
        for jp in nbrs:
            pre_t += W[j, jp] * controllers[jp].theta_time
            pre_w += W[j, jp] * controllers[jp].theta_wait

        # Проекция Π_{X_i} — немного разные границы у каждого контроллера
        lo_t = 0.3 + 0.05 * cid
        hi_t = 12.0 + 0.5 * cid
        lo_w = 0.1 + 0.03 * cid
        hi_w = 6.0 + 0.2 * cid
        nt = np.clip(pre_t, lo_t, hi_t)
        nw = np.clip(pre_w, lo_w, hi_w)

        S[cid]["pg_t"] = g_t.copy()
        S[cid]["pg_w"] = g_w.copy()
        S[cid]["y_t"]  = y_t_new
        S[cid]["y_w"]  = y_w_new

        new_t.append(nt)
        new_w.append(nw)

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = t
        ctrl.theta_wait = w


# =============================================================================
# 6. Primal-Dual Bandit — двухточечный
#    Yi, Li, Yang, Xie, Chai, Johansson, IEEE TAC 2021
# =============================================================================

def pd_bandit_twopoint(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
    eta: float = 0.05,
) -> None:
    """
    Bandit OCO с двухточечной обратной связью и парным ограничением.

    Ограничение: mean(θ_time) ≤ C_budget (мягкий бюджет очереди).
    Primal: θ ← Π[θ − α·(ĝ + λ·∇h) + consensus]
    Dual:   λ ← max(0, λ + η·h(θ))
    Regret: O(√T), constraint violation: O(T^{3/4}).
    """
    S = _st("pd2")
    B = _comm_graph(step, len(controllers))
    alpha_t = _decay_alpha(step, alpha)
    sigma   = _decay_pert(step, beta)
    logger.debug("[PD-2pt] step=%d  alpha_t=%.5f  sigma=%.5f", step, alpha_t, sigma)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "pd_2pt", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(sigma)})
    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        cid = ctrl.id
        dim = len(ctrl.theta_time)

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        if cid not in S:
            S[cid] = {"lam": 0.0}

        lam = S[cid]["lam"]

        # Двухточечный сферический оценщик
        u_t = np.random.randn(dim); u_t /= np.linalg.norm(u_t) + 1e-8
        u_w = np.random.randn(dim); u_w /= np.linalg.norm(u_w) + 1e-8

        g_t = (dim / (2 * sigma)) * (
            _proc_loss(ctrl.theta_time + sigma * u_t, ctrl.obs_history) -
            _proc_loss(ctrl.theta_time - sigma * u_t, ctrl.obs_history)
        ) * u_t

        g_w = (dim / (2 * sigma)) * (
            _wait_loss(ctrl.theta_wait + sigma * u_w, ctrl.obs_history) -
            _wait_loss(ctrl.theta_wait - sigma * u_w, ctrl.obs_history)
        ) * u_w

        c_t = _consensus(B, j, controllers, "theta_time")
        c_w = _consensus(B, j, controllers, "theta_wait")

        # Primal: gradient + constraint + consensus
        num_types = getattr(ctrl, "num_types", 5)
        nt = ctrl.theta_time - alpha_t * (g_t + lam * _grad_h(ctrl.theta_time, num_types) + gamma * c_t)
        nw = ctrl.theta_wait - alpha_t * (g_w + gamma * c_w)

        # Dual update
        S[cid]["lam"] = max(0.0, lam + eta * _h(ctrl.theta_time, num_types))

        new_t.append(nt)
        new_w.append(nw)

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)


# =============================================================================
# 7. Primal-Dual Bandit — одноточечный
#    Yi, Li, Yang, Xie, Chai, Johansson, IEEE TAC 2021
# =============================================================================

def pd_bandit_onepoint(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
    eta: float = 0.05,
) -> None:
    """
    Bandit OCO с одноточечной обратной связью.

    Один вызов функции на контроллер — минимальный oracle.
    Regret: O(T^{5/6}) — медленнее двухточечного, но дешевле.
    """
    S = _st("pd1")
    B = _comm_graph(step, len(controllers))
    alpha_t = _decay_alpha(step, alpha)
    sigma   = _decay_pert(step, beta)
    logger.debug("[PD-1pt] step=%d  alpha_t=%.5f  sigma=%.5f", step, alpha_t, sigma)
    if _TRACE_ENABLED:
        _TRACE_LOG.append({"step": step, "variant": "pd_1pt", "alpha_k": float(alpha_t), "kappa_k": float("nan"), "pert": float(sigma)})
    new_t, new_w = [], []

    for j, ctrl in enumerate(controllers):
        cid = ctrl.id
        dim = len(ctrl.theta_time)

        if len(ctrl.obs_history) < 10:
            new_t.append(ctrl.theta_time.copy())
            new_w.append(ctrl.theta_wait.copy())
            continue

        if cid not in S:
            S[cid] = {"lam": 0.0}

        lam = S[cid]["lam"]

        u_t = np.random.randn(dim); u_t /= np.linalg.norm(u_t) + 1e-8
        u_w = np.random.randn(dim); u_w /= np.linalg.norm(u_w) + 1e-8

        # Одноточечный оценщик: (d/σ)·f(θ+σu)·u
        g_t = (dim / sigma) * _proc_loss(ctrl.theta_time + sigma * u_t, ctrl.obs_history) * u_t
        g_w = (dim / sigma) * _wait_loss(ctrl.theta_wait + sigma * u_w, ctrl.obs_history) * u_w

        c_t = _consensus(B, j, controllers, "theta_time")
        c_w = _consensus(B, j, controllers, "theta_wait")

        num_types = getattr(ctrl, "num_types", 5)
        nt = ctrl.theta_time - alpha_t * (g_t + lam * _grad_h(ctrl.theta_time, num_types) + gamma * c_t)
        nw = ctrl.theta_wait - alpha_t * (g_w + gamma * c_w)

        S[cid]["lam"] = max(0.0, lam + eta * _h(ctrl.theta_time, num_types))

        new_t.append(nt)
        new_w.append(nw)

    for ctrl, t, w in zip(controllers, new_t, new_w):
        ctrl.theta_time = _clip_t(t)
        ctrl.theta_wait = _clip_w(w)


# =============================================================================
# Ablation 1 — A-SPSA без моментума (β_nes ≡ 0)
#   Изолирует вклад Nesterov-ускорения: y_t → θ_t, momentum отключён.
#   Эквивалентно SPSA с тем же градиентом в точке y (но y=θ, так что идентично).
# =============================================================================

def aspsa_no_momentum(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
    beta_nes_max: float = 1.0,
) -> None:
    """Ablation: α_k≡0 → x̃=θ̄, z̄ frozen — pure SPSA descent, no surrogate."""
    h = alpha; mu = 0.5 * h; eta = 0.1 * h
    _aspsa_step(_st("aspsa_no_momentum"), "aspsa_no_momentum",
                controllers, step, h, beta, gamma, mu, eta, fixed_alpha=0.0,
                alpha_cap=beta_nes_max)


# =============================================================================
# Ablation 2 — A-SPSA с фиксированным β=0.5 (не адаптивным)
#   Проверяет, важен ли адаптивный schedule β_t=(t-1)/(t+2)
#   против постоянного тяжёлого шарика.
# =============================================================================

def aspsa_fixed_beta(
    controllers, step: int,
    alpha: float = 0.01, beta: float = 0.1, gamma: float = 0.02,
    beta_nes_max: float = 1.0,
) -> None:
    """Ablation: α_k≡0.5 (fixed, not adaptive) — surrogate active but schedule frozen."""
    h = alpha; mu = 0.5 * h; eta = 0.1 * h
    _aspsa_step(_st("aspsa_fixed_beta"), "aspsa_fixed_beta",
                controllers, step, h, beta, gamma, mu, eta, fixed_alpha=0.5,
                alpha_cap=beta_nes_max)


# =============================================================================
# Реестр — единая точка доступа
# =============================================================================

VARIANTS: Dict[str, Any] = {
    "spsa":              None,              # оригинал в main.py
    "aspsa":             aspsa_consensus,
    "aspsa_no_momentum": aspsa_no_momentum,
    "aspsa_fixed_beta":  aspsa_fixed_beta,
    "kw":                kw_consensus,
    "zo_pgd":            zo_pgd_consensus,
    "sp_gt":             sp_gt_consensus,
    "zo_gt":             zo_gt_consensus,
    "pd_2pt":            pd_bandit_twopoint,
    "pd_1pt":            pd_bandit_onepoint,
}

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "spsa":              "SPSA + consensus (Радемахер δ, 2 вызова)",
    "aspsa":             "A-SPSA + Нестеров (IEEE 2025, 2 вызова)",
    "aspsa_no_momentum": "A-SPSA, β≡0 (без моментума, абляция)",
    "aspsa_fixed_beta":  "A-SPSA, β=0.5 фикс. (не адаптивный, абляция)",
    "kw":                "Kiefer-Wolfowitz (покоординатный, 2d вызовов)",
    "zo_pgd":            "ZO Projected GD, сферическое сглаживание (NeurIPS 2021)",
    "sp_gt":             "Single-Point Gradient Tracking (ICML 2023, 1 вызов)",
    "zo_gt":             "ZO Gradient Tracking + проекция (IFAC 2023)",
    "pd_2pt":            "Primal-Dual Bandit, 2 точки (IEEE TAC 2021, с ограничением)",
    "pd_1pt":            "Primal-Dual Bandit, 1 точка (IEEE TAC 2021, с ограничением)",
}


def call_variant(
    name: str,
    controllers,
    step: int,
    alpha: float = 0.01,
    beta: float = 0.1,
    gamma: float = 0.02,
    beta_nes_max: float = 1.0,
) -> None:
    """Вызвать выбранный вариант по имени. 'spsa' — передаётся main.py."""
    fn = VARIANTS.get(name)
    if fn is not None:
        if name in ("aspsa", "aspsa_no_momentum", "aspsa_fixed_beta"):
            fn(controllers, step, alpha=alpha, beta=beta, gamma=gamma, beta_nes_max=beta_nes_max)
        else:
            fn(controllers, step, alpha=alpha, beta=beta, gamma=gamma)

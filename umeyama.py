#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""刚体配准 Umeyama(尺度固定 1)= Kabsch:求 R,t 使 Q ≈ R·P + t。SVD + 反射保护。
   外加 correspond():点对应未知时,最近邻指派迭代找对应(避开 N! 暴力)。"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def umeyama(P, Q):
    """P,Q: (N,3) 一一对应点。返回 R(3,3), t(3,), rms(米)。

    R 把 P 映到 Q(列主序):pred_i = R·P_i + t。SVD 反射保护防镜像(det<0)。
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"P/Q 形状需相同且为 (N,3),实为 {P.shape} / {Q.shape}")
    if len(P) < 3:
        raise ValueError(f"点数不足(<3):{len(P)}")
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)                       # 3×3 互协方差 Σ p_i q_iᵀ
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0   # 退化(共面/共线 det=0)取 +1
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - R @ cP
    rms = float(np.sqrt(np.mean(np.sum((Q - (P @ R.T + t)) ** 2, axis=1))))
    return R, t, rms


def residuals(P, Q, R, t):
    """每点 ‖(R·p+t) − q‖(米)。"""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    pred = P @ R.T + t
    return np.linalg.norm(pred - Q, axis=1)


def correspond(P, Q, iters=5):
    """点对应未知时,迭代最近邻指派找一一对应(N 点 O(N³),避开 N! 暴力)。

    返回 (ri, ci):P[ri] 与 Q[ci] 一一对应。需 len(P)==len(Q)。
    先按同序起手算 R,t,再反复"指派→重解"。多数点本就同序时一两轮即收敛。
    """
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(P) != len(Q):
        raise ValueError(f"点数不等 {len(P)} vs {len(Q)}")
    R, t, _ = umeyama(P, Q)
    ri = ci = np.arange(len(P))
    for _ in range(iters):
        D = np.linalg.norm((P @ R.T + t)[:, None] - Q[None], axis=2)
        ri, ci = linear_sum_assignment(D)
        R, t, _ = umeyama(P[ri], Q[ci])
    return ri, ci

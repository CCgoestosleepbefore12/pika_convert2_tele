#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定流程入口:单条录制的 pika 点 + robot 点 → ᴮʳT_P0 + 留一交叉验证(LOO)+ 可选丢离群点 + 出图。

核心数学在 calib_tip_to_base / umeyama;这里只串流程。
输入两个点 JSON(dwell_points.py 产):
  --pika  {"P0":..,"points":[N×pose]}      pika 角点(基站系)
  --robot {"d":米,"points":[N×pose]}        robot 角点(法兰,robot base)
对应未知用 correspond() 自动匹配(不靠时间序)。--drop_worst 丢掉残差最大的一个点重标
(适合某角点 pika/robot 碰歪了 ~几cm 的情况)。

精度判读:看 **LOO**(留一,每点预测时不参与自己的标定)= 真实泛化精度,而非"拟合"(乐观)。

用法:
  python calibrate_arm.py --pika pika_l.json --robot robot_l.json --arm l --drop_worst \
      --out_T T_Br_P0_l.json --out_png left_calib.png
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_tip_to_base import pika_positions_in_P0, robot_tcp_positions, planarity_ratio, verdict
from umeyama import umeyama, residuals, correspond
from pose_utils import atomic_dump_json


def _loo(P, Q):
    """留一交叉验证:逐点用其余点标定再预测自己。返回逐点误差(mm)。"""
    out = []
    for i in range(len(P)):
        idx = [k for k in range(len(P)) if k != i]
        R, t, _ = umeyama(P[idx], Q[idx])
        out.append(np.linalg.norm(P[i] @ R.T + t - Q[i]) * 1000)
    return np.array(out)


def calibrate_arm(pika, robot, drop_worst=False):
    """返回 dict:T_Br_P0、拟合/ LOO 残差、对应、共面比、(可选)丢弃点。"""
    P = pika_positions_in_P0(pika)
    Q = robot_tcp_positions(robot)
    ri, ci = correspond(P, Q)                     # 自动找一一对应
    P, Q = P[ri], Q[ci]
    R, t, _ = umeyama(P, Q)
    res = residuals(P, Q, R, t) * 1000
    dropped = None
    if drop_worst and len(P) > 4:
        dropped = int(np.argmax(res))
        keep = [k for k in range(len(P)) if k != dropped]
        P, Q = P[keep], Q[keep]
        R, t, _ = umeyama(P, Q)
        res = residuals(P, Q, R, t) * 1000
    loo = _loo(P, Q)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return {"T_Br_P0": T, "P": P, "Q": Q, "res": res, "loo": loo, "perm_pika": ri.tolist(),
            "perm_robot": ci.tolist(), "planarity": planarity_ratio(P), "dropped": dropped,
            "d": robot["d"]}


def plot(rep, arm, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P, Q, res, loo = rep["P"], rep["Q"], rep["res"], rep["loo"]
    Pa = P @ rep["T_Br_P0"][:3, :3].T + rep["T_Br_P0"][:3, 3]
    n = len(P)
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(*Q.T, c="red", s=60, label="robot TCP", depthshade=False)
    ax.scatter(*Pa.T, c="blue", s=40, label="pika->robot", depthshade=False)
    for i in range(n):
        ax.plot(*zip(Q[i], Pa[i]), c="0.4", lw=1)
        ax.text(Q[i, 0], Q[i, 1], Q[i, 2], f"  {res[i]:.0f}", fontsize=9)
    ax.set_title(f"arm {arm} {n}pts d={rep['d']*100:.1f}  fit {res.mean():.1f}/{res.max():.1f}mm")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.legend(fontsize=8)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(n)
    ax2.bar(x - 0.2, res, 0.4, color="steelblue", label="fit")
    ax2.bar(x + 0.2, loo, 0.4, color="orange", label="LOO")
    ax2.axhline(8, color="green", ls="--", lw=1, label="8mm")
    ax2.set_xlabel("point"); ax2.set_ylabel("mm")
    ax2.set_title(f"LOO mean {loo.mean():.1f}/max {loo.max():.1f}mm")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_png, dpi=110)


def main():
    ap = argparse.ArgumentParser(description="单条录制点 → ᴮʳT_P0 + LOO + 出图")
    ap.add_argument("--pika", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--drop_worst", action="store_true", help="丢掉残差最大的一个点重标(某角点碰歪时)")
    ap.add_argument("--out_T", default="T_Br_P0.json")
    ap.add_argument("--out_png", default=None)
    a = ap.parse_args()
    rep = calibrate_arm(json.load(open(a.pika)), json.load(open(a.robot)), a.drop_worst)
    res, loo = rep["res"], rep["loo"]
    out = {"T_Br_P0": rep["T_Br_P0"].tolist(), "arm": a.arm, "d": rep["d"],
           "fit_mm": res.tolist(), "loo_mm": loo.tolist(), "planarity": rep["planarity"],
           "perm_pika": rep["perm_pika"], "perm_robot": rep["perm_robot"], "dropped": rep["dropped"]}
    atomic_dump_json(out, a.out_T)
    tag = f"(丢了#{rep['dropped']})" if rep["dropped"] is not None else ""
    print(f"[臂 {a.arm}] {len(rep['P'])}点{tag} d={rep['d']*100:.1f}cm 共面比 {rep['planarity']:.3f}")
    print(f"  拟合 mean {res.mean():.1f}/max {res.max():.1f}mm → {verdict(res.mean())}")
    print(f"  LOO  mean {loo.mean():.1f}/max {loo.max():.1f}mm  ← 真实精度 → {verdict(loo.mean())}")
    print(f"  已存 {a.out_T}")
    if a.out_png:
        plot(rep, a.arm, a.out_png)
        print(f"  图 {a.out_png}")


if __name__ == "__main__":
    main()

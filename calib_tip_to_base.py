#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四点标定:求 ᴮʳT_P0(pika 第一帧/固定工装 → robot base)。放置版(只用位置)。

pika 侧:ᴾ⁰T_Pj = (ᴮᵖT_P0)⁻¹·ᴮᵖT_Pj,取尖端位置 p_j = trans(·)(前提:pika raw 就是尖端)
robot 侧:ᴮʳT_Gj = ᴮʳT_Fj·ᶠT_G(法兰→TCP,沿法兰 +z 偏 d),取 q_j = trans(·)
配准:q_j ≈ R·p_j + t(Umeyama s=1,SVD)→ ᴮʳT_P0 = [R|t]

输入(dwell_points.py 产出):
  --pika  {"P0":pose, "points":[pose×N]}     pose 欧拉或四元数(pika 基站系)
  --robot {"d":米, "points":[pose×N]}        pose 四元数(robot base 法兰)
  可选 --pika_val/--robot_val:留出验证点(另一刚体,不参与标定,只报误差)
输出:--out ᴮʳT_P0 的 JSON + 同名 .report.txt
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import make_T, inv_T, pose_dict_to_T, atomic_dump_json  # noqa: E402
from umeyama import umeyama, residuals, correspond                     # noqa: E402


def pika_positions_in_P0(pika):
    """rebase pika 标定点到 P0,返回尖端位置 (N,3)(P0 系)。"""
    T_P0_Bp = inv_T(pose_dict_to_T(pika["P0"]))
    return np.array([(T_P0_Bp @ pose_dict_to_T(p))[:3, 3] for p in pika["points"]])


def robot_tcp_positions(robot):
    """法兰位姿 + d → TCP 尖端位置 (N,3)(robot base)。"""
    T_F_G = make_T(np.eye(3), [0.0, 0.0, robot["d"]])
    return np.array([(pose_dict_to_T(p) @ T_F_G)[:3, 3] for p in robot["points"]])


def _spread(P):
    """点间最小距离(mm),判标定稳健性。"""
    n = len(P)
    return min(np.linalg.norm(P[i] - P[j]) for i in range(n) for j in range(i + 1, n)) * 1000


def planarity_ratio(P):
    """中心化点的 最小/最大 奇异值比;→0 说明近共面/共线,绕该轴旋转欠约束(标定病态)。"""
    s = np.linalg.svd(np.asarray(P, float) - np.asarray(P, float).mean(0), compute_uv=False)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def _best_perm(P, Q):
    """correspond 找 P↔Q 一一对应,返回 P 行排列 perm 使 P[perm] 对齐 Q(Q 固定)。避开 N! 暴力。"""
    if len(P) < 3:
        return np.arange(len(P))            # 点太少(correspond 需 ≥3),退化为同序
    ri, ci = correspond(P, Q)
    perm = np.empty(len(P), int)
    perm[ci] = ri
    return perm


def calibrate(pika, robot, val_pika=None, val_robot=None, auto_corr=True):
    """返回 (T_Br_P0 4×4, report dict)。auto_corr:自动搜 pika↔robot 最佳对应(不靠时间序)。"""
    P = pika_positions_in_P0(pika)
    Q = robot_tcp_positions(robot)
    if len(P) != len(Q):
        raise ValueError(f"点数不等:pika {len(P)} vs robot {len(Q)}")
    perm = _best_perm(P, Q) if auto_corr else np.arange(len(P))
    P = P[perm]
    R, t, rms = umeyama(P, Q)
    T_Br_P0 = make_T(R, t)
    res = residuals(P, Q, R, t)
    rep = {"n_points": int(len(P)), "corr_perm": perm.tolist(), "per_point_mm": (res * 1000).tolist(),
           "mean_mm": float(res.mean() * 1000), "max_mm": float(res.max() * 1000),
           "rms_mm": rms * 1000, "min_point_dist_mm": _spread(P),
           "planarity_ratio": planarity_ratio(P), "tcp_offset_d_m": robot["d"]}
    if val_pika and val_robot:
        Pv, Qv = pika_positions_in_P0(val_pika), robot_tcp_positions(val_robot)
        vperm = _best_perm(Pv, Qv) if auto_corr else np.arange(len(Pv))
        Pv = Pv[vperm]
        vres = residuals(Pv, Qv, R, t)
        rep["val_corr_perm"] = vperm.tolist()
        rep["val_per_point_mm"] = (vres * 1000).tolist()
        rep["val_mean_mm"] = float(vres.mean() * 1000)
        rep["val_max_mm"] = float(vres.max() * 1000)
    return T_Br_P0, rep


def verdict(e):
    return ("很好" if e < 3 else "可用" if e < 8 else
            "需检查(TCP/接触/顺序/共面)" if e < 20 else "大概率有错(坐标系/TCP/顺序/单位)")


def write_outputs(T_Br_P0, rep, out_path, meta):
    atomic_dump_json({"T_Br_P0": T_Br_P0.tolist(), "residual": rep, "meta": meta}, out_path)
    rpt = out_path.rsplit(".", 1)[0] + ".report.txt"
    L = [f"四点标定报告 ᴮʳT_P0(放置版,arm={meta.get('arm','?')})",
         f"点数 {rep['n_points']}  TCP偏移 d={rep['tcp_offset_d_m']}m  点间最小距 {rep['min_point_dist_mm']:.0f}mm  "
         f"共面比 {rep['planarity_ratio']:.3f}{'(⚠近共面)' if rep['planarity_ratio'] < 0.1 else ''}",
         "标定点残差(R·p+t vs q):"]
    L += [f"  point {i}: {e:.1f} mm" for i, e in enumerate(rep["per_point_mm"])]
    L.append(f"  mean {rep['mean_mm']:.1f}  max {rep['max_mm']:.1f}  rms {rep['rms_mm']:.1f} mm → {verdict(rep['mean_mm'])}")
    if "val_mean_mm" in rep:
        L.append("留出验证点(不参与标定):")
        L += [f"  val {i}: {e:.1f} mm" for i, e in enumerate(rep["val_per_point_mm"])]
        L.append(f"  val mean {rep['val_mean_mm']:.1f}  max {rep['val_max_mm']:.1f} mm → {verdict(rep['val_mean_mm'])}")
    with open(rpt, "w") as f:
        f.write("\n".join(L) + "\n")
    return rpt


def main():
    ap = argparse.ArgumentParser(description="四点标定 ᴮʳT_P0(pika P0 → robot base)")
    ap.add_argument("--pika", required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--pika_val", default=None)
    ap.add_argument("--robot_val", default=None)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--no_auto_corr", action="store_true", help="关掉自动搜最佳对应(默认开:不靠时间序)")
    ap.add_argument("--out", default="T_Br_P0.json")
    a = ap.parse_args()
    pika, robot = json.load(open(a.pika)), json.load(open(a.robot))
    vp = json.load(open(a.pika_val)) if a.pika_val else None
    vr = json.load(open(a.robot_val)) if a.robot_val else None
    T_Br_P0, rep = calibrate(pika, robot, vp, vr, auto_corr=not a.no_auto_corr)
    meta = {"arm": a.arm, "src_pika": os.path.abspath(a.pika), "src_robot": os.path.abspath(a.robot),
            "unit": "meter", "method": "Umeyama(s=1) on N tip points + auto correspondence", "note": "position-only; output orientation = pika convention (no Y)"}
    rpt = write_outputs(T_Br_P0, rep, a.out, meta)
    print(f"已保存 {a.out} 和 {rpt}")
    print(f"  对应 cali={rep['corr_perm']}" + (f" verify={rep.get('val_corr_perm')}" if 'val_corr_perm' in rep else ""))
    tag = f"  验证点 mean {rep['val_mean_mm']:.1f}mm" if "val_mean_mm" in rep else ""
    print(f"残差 mean {rep['mean_mm']:.1f}mm max {rep['max_mm']:.1f}mm 共面比 {rep['planarity_ratio']:.3f} → {verdict(rep['mean_mm'])}{tag}")
    if rep["planarity_ratio"] < 0.1:
        print(f"  ⚠ 四点近共面(共面比 {rep['planarity_ratio']:.3f}<0.1):绕面内轴旋转欠约束,离面轨迹会偏;"
              "且同面验证点会假合格。请让 4 点带明显高度差、验证点跨面。")
    if rep["mean_mm"] > 8:
        print("  ⚠ 残差偏大:查 点顺序一致? pika 真是尖端? robot 已用 eef(法兰)+d? d 方向(法兰+z)? 单位? 四点近共面?")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健提停留点(位置稳定 + 真实时长),对手持抖动鲁棒,取代旧的"最长 N"启发式。

停留 = 在 ±win 窗内位置散布 <R、且连续时长 ≥min_dur 的段;空间相邻段去重。
两种源:
  --pika <packed.hdf5> --group <名>   pika 打包文件(pack_poses.py 产);时间用真实秒,win/min_dur 单位=秒
  --tele <eef.hdf5> --arm l/r         tele eef_quaternion(法兰);时间用帧序,win/min_dur 单位=帧
输出标定点 JSON:
  pika  -> {"kind":"pika","P0":首帧, "points":[N×pose]}(⚠ 录制须从工装起步,P0=首帧)
  tele  -> {"kind":"robot","d":米, "points":[N×pose]}

用法:
  python dwell_points.py --pika pika_packed.hdf5 --group pika_l_cali/pika_l --n 4 --out pika_l.json
  python dwell_points.py --tele tele_l_cali.hdf5 --arm l --d 0.138 --n 4 --out robot_l.json
"""
import os
import sys
import argparse

import numpy as np
import h5py
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import atomic_dump_json  # noqa: E402


def find_dwells(ts, pos, quat, R, win, min_dur, merge):
    """位置稳定+时长 → 全部停留段(按时间序)。返回 [(pos均值,quat均值,(s,e),时长), ...],n_total。
    选最长 N 个/排序的事交给调用方。"""
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise ValueError("pos/quat 含 NaN/Inf,先清洗")
    m = len(ts)
    lo = np.searchsorted(ts, ts - win)
    hi = np.searchsorted(ts, ts + win)
    parked = np.array([hi[i] > lo[i] and
                       np.linalg.norm(pos[lo[i]:hi[i]] - pos[lo[i]:hi[i]].mean(0), axis=1).max() < R
                       for i in range(m)])
    runs, i = [], 0
    while i < m:
        if parked[i]:
            j = i
            while j < m and parked[j]:
                j += 1
            if ts[j - 1] - ts[i] >= min_dur:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for s, e in runs:
        p = pos[s:e].mean(0)
        q = Rot.from_quat(quat[s:e]).mean().as_quat()
        if out and np.linalg.norm(p - out[-1][0]) < merge:    # 同一停留的碎片,合并
            continue
        out.append((p, q, (s, e), float(ts[e - 1] - ts[s])))
    return out, len(out)


def _pose(p, q):
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3])}


def _select(dw, n, skip_first):
    """按时间排序 → 可选丢最早一段(起点)→ 取最长 n → 按时间排。"""
    dw = sorted(dw, key=lambda d: d[2][0])      # 时间序
    if skip_first and dw:
        dw = dw[1:]
    return sorted(sorted(dw, key=lambda d: d[3], reverse=True)[:n], key=lambda d: d[2][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pika", help="打包 hdf5(pack_poses 产)")
    ap.add_argument("--group", help="(pika)组名,如 pika_l_cali/pika_l")
    ap.add_argument("--tele", help="tele eef_quaternion hdf5")
    ap.add_argument("--arm", choices=["l", "r"], help="(tele)臂")
    ap.add_argument("--n", type=int, default=4, help="刚体角点数")
    ap.add_argument("--d", type=float, default=0.138, help="(tele)法兰→尖端沿 +z(米)")
    ap.add_argument("--R", type=float, default=0.04, help="停留判定半径(米);手持大可调大")
    ap.add_argument("--win", type=float, default=None, help="窗(pika 秒/tele 帧);默认 pika0.5 tele15")
    ap.add_argument("--min_dur", type=float, default=None, help="最短停留(pika 秒/tele 帧);默认 pika3.5 tele40")
    ap.add_argument("--merge", type=float, default=0.05, help="去重:相邻停留中心距阈(米)")
    ap.add_argument("--skip_first", action="store_true", help="丢掉时间最早的停留段(起点/工装),只留角点")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.pika:
        with h5py.File(a.pika, "r") as h:
            g = h[a.group]
            ts = np.asarray(g["t"][:], float)
            pos = np.asarray(g["pos"][:], float)
            quat = Rot.from_euler("xyz", np.asarray(g["rpy"][:], float)).as_quat()
        win = 0.5 if a.win is None else a.win
        md = 3.5 if a.min_dur is None else a.min_dur
        dw, ntot = find_dwells(ts, pos, quat, a.R, win, md, a.merge)
        dw = _select(dw, a.n, a.skip_first)
        out = {"kind": "pika", "P0": _pose(pos[0], quat[0]), "points": [_pose(p, q) for p, q, _, _ in dw],
               "n_found": len(dw), "n_total": ntot}
    elif a.tele:
        with h5py.File(a.tele, "r") as h:
            eq = np.asarray(h["observations/eef_quaternion"][:], float)
        c = 0 if a.arm == "l" else 8
        pos, quat = eq[:, c:c + 3], eq[:, c + 3:c + 7]
        ts = np.arange(len(pos)).astype(float)
        win = 15 if a.win is None else a.win
        md = 40 if a.min_dur is None else a.min_dur
        dw, ntot = find_dwells(ts, pos, quat, min(a.R, 0.012), win, md, a.merge)
        dw = _select(dw, a.n, a.skip_first)
        out = {"kind": "robot", "arm": a.arm, "d": a.d, "points": [_pose(p, q) for p, q, _, _ in dw],
               "n_found": len(dw), "n_total": ntot}
    else:
        raise SystemExit("需 --pika 或 --tele")

    atomic_dump_json(out, a.out)
    print(f"检出停留 {out['n_total']} 段,取 {out['n_found']}/{a.n} → {a.out}")
    for k, (p, q, (s, e), dur) in enumerate(dw):
        print(f"  pt{k} {dur:.1f}{'s' if a.pika else 'f'} 帧[{s}:{e}] pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
    if out["n_total"] != a.n:
        print(f"  ⚠ 停留段 {out['n_total']}≠{a.n}:核对选中点;可调 --min_dur/--R/--merge")


if __name__ == "__main__":
    main()

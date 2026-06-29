#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 pika 轨迹转换到 robot base(放置版)。支持单臂 pose 目录/json,以及 synced hdf5 双臂。

ᴮʳT_Pi = ᴮʳT_P0 · (ᴮᵖT_P0)⁻¹ · ᴮᵖT_Pi
  ᴮʳT_P0:calibrate_arm 标出的常量(T_Br_P0_{l,r}.json)
  ᴮᵖT_P0:**这条 episode 自己的第一帧**(每条用各自第一帧 rebase;前提:都从同一固定工装起步)

输入(--traj):
  · pose json 目录(每帧 {x,y,z,roll,pitch,yaw},时间戳命名)        → 单臂,用 --calib
  · 单 json {"frames":[<pose>,...]}                                  → 单臂,用 --calib
  · synced hdf5(sync_umi_raw 产,observations/pose_left|right (T,6) 欧拉,基站系)
                                                                      → 双臂,用 --calib_l --calib_r
输出(--out .json):
  单臂 {"frames":[{x,y,z,qx,qy,qz,qw},...]}
  双臂 {"left":[...],"right":[...]}     + 附存 .npy(单臂 (T,7) / 双臂 (T,14)=[左7,右7])
  (robot base;位置=尖端;朝向为 pika 约定、未做 Y)
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import inv_T, pose_dict_to_T, T_to_pose_dict, load_pose_dir, atomic_dump_json  # noqa: E402


def load_traj(path):
    """pose 目录 或 单 json({"frames":[...]})→ pose dict 列表(单臂,欧拉)。"""
    if os.path.isdir(path):
        return load_pose_dir(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--traj 既不是目录也不是文件:{path}")
    frames = json.load(open(path)).get("frames")
    if not frames:
        raise ValueError(f"{path} 没有非空 'frames'")
    return frames


def load_arm_hdf5(path, arm):
    """synced hdf5 → 指定臂的 pose dict 列表。observations/pose_{left,right} (T,6) 欧拉,基站系。"""
    import h5py
    key = "observations/pose_left" if arm == "l" else "observations/pose_right"
    with h5py.File(path, "r") as h:
        pose = np.asarray(h[key][:], float)
    if pose.ndim != 2 or pose.shape[1] != 6:
        raise ValueError(f"{key} 形状 {pose.shape} 异常(应为 T×6 欧拉)")
    return [{"x": p[0], "y": p[1], "z": p[2], "roll": p[3], "pitch": p[4], "yaw": p[5]} for p in pose]


def convert(frames, T_Br_P0):
    """逐帧 rebase(用 frames[0] 作本条 ᴮᵖT_P0)+ 左乘 ᴮʳT_P0。返回 (T,4,4)。"""
    T_Br_P0 = np.asarray(T_Br_P0, float)
    T_P0_Bp = inv_T(pose_dict_to_T(frames[0]))
    return np.array([T_Br_P0 @ T_P0_Bp @ pose_dict_to_T(d) for d in frames])


def _load_T(path):
    return np.array(json.load(open(path))["T_Br_P0"], float)


def _row(p):
    return [p["x"], p["y"], p["z"], p["qx"], p["qy"], p["qz"], p["qw"]]


def main():
    ap = argparse.ArgumentParser(description="pika 轨迹 → robot base(单臂目录/json 或 synced hdf5 双臂)")
    ap.add_argument("--traj", required=True, help="pose 目录 / 单 json / synced hdf5")
    ap.add_argument("--calib", help="单臂:T_Br_P0.json")
    ap.add_argument("--calib_l", help="hdf5 双臂:左臂 T_Br_P0_l.json")
    ap.add_argument("--calib_r", help="hdf5 双臂:右臂 T_Br_P0_r.json")
    ap.add_argument("--out", default="traj_robotbase.json")
    a = ap.parse_args()
    npy = a.out.rsplit(".", 1)[0] + ".npy"
    note = "position = tip in robot base; orientation in pika convention (no Y)"

    if a.traj.endswith(".hdf5"):                                   # synced hdf5 → 双臂
        if not (a.calib_l and a.calib_r):
            raise SystemExit("hdf5 双臂需 --calib_l 和 --calib_r")
        Tl, Tr = _load_T(a.calib_l), _load_T(a.calib_r)
        L = [T_to_pose_dict(T) for T in convert(load_arm_hdf5(a.traj, "l"), Tl)]
        R = [T_to_pose_dict(T) for T in convert(load_arm_hdf5(a.traj, "r"), Tr)]
        atomic_dump_json({"frame_id": "robot_base", "n": len(L), "left": L, "right": R, "note": note}, a.out)
        np.save(npy, np.array([_row(l) + _row(r) for l, r in zip(L, R)]))   # (T,14)=[左7,右7]
        print(f"已保存 {a.out} 和 .npy(双臂 {len(L)} 帧,(T,14))")
    else:                                                          # 单臂目录/json
        if not a.calib:
            raise SystemExit("单臂需 --calib")
        poses = [T_to_pose_dict(T) for T in convert(load_traj(a.traj), _load_T(a.calib))]
        atomic_dump_json({"frame_id": "robot_base", "n": len(poses), "frames": poses, "note": note}, a.out)
        np.save(npy, np.array([_row(p) for p in poses]))
        print(f"已保存 {a.out} 和 .npy(单臂 {len(poses)} 帧)")


if __name__ == "__main__":
    main()

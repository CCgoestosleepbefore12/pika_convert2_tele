#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""位姿工具(numpy/scipy):欧拉/四元数 ↔ 4×4,刚体求逆。

约定(与本项目一致):pika 欧拉 [roll,pitch,yaw] 固定轴 xyz = scipy from_euler('xyz');四元数 xyzw。
"""
import os
import json
import glob

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def make_T(R, t):
    """3×3 旋转 + 3 平移 → 4×4。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    """刚体逆 [R|t]⁻¹ = [Rᵀ | -Rᵀt]。"""
    R = T[:3, :3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def euler_to_R(rpy):
    """[roll,pitch,yaw] 固定轴 xyz → 3×3。"""
    return Rot.from_euler("xyz", rpy).as_matrix()


def quat_to_R(quat_xyzw):
    """四元数 xyzw → 3×3。"""
    return Rot.from_quat(quat_xyzw).as_matrix()


def pose_dict_to_T(d):
    """{x,y,z, roll,pitch,yaw}(欧拉)或 {x,y,z, qx,qy,qz,qw}(四元数)→ 4×4。"""
    t = np.array([d["x"], d["y"], d["z"]], float)
    if "qw" in d:
        R = quat_to_R([d["qx"], d["qy"], d["qz"], d["qw"]])
    else:
        R = euler_to_R([d["roll"], d["pitch"], d["yaw"]])
    return make_T(R, t)


def T_to_pose_dict(T):
    """4×4 → {x,y,z, qx,qy,qz,qw}(输出统一四元数)。"""
    q = Rot.from_matrix(T[:3, :3]).as_quat()   # xyzw
    t = T[:3, 3]
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
            "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]), "qw": float(q[3])}


def load_pose_dir(d):
    """目录下每帧一个 pose json,按文件名(时间戳)升序 → [dict, ...]。非纯数字名跳过。"""
    fs = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            fs.append((float(os.path.splitext(os.path.basename(f))[0]), f))
        except ValueError:
            continue   # 跳过 meta.json 等非时间戳文件
    if not fs:
        raise ValueError(f"{d} 下没有时间戳命名的 *.json")
    return [json.load(open(f)) for _, f in sorted(fs)]


def atomic_dump_json(obj, path):
    """原子落盘 JSON(.tmp + os.replace),与仓库 dump_dp_hdf5 习惯一致。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 pika pose json 目录树一次性打包成单个 hdf5(加速反复读取;原本逐 json 读上万个很慢)。

每个含"时间戳命名 *.json"的叶子目录 → hdf5 里一个组,含:
  t   (T,)   时间戳(秒,文件名)
  pos (T,3)  x,y,z
  rpy (T,3)  roll,pitch,yaw(固定轴 xyz)
组名 = 该目录相对 --src 的路径(如 pika_l_cali/pika_l)。

用法:python pack_poses.py --src <pika_ori 根> --out pika_packed.hdf5
"""
import os
import glob
import json
import argparse

import numpy as np
import h5py


def _ts(f):
    return float(os.path.splitext(os.path.basename(f))[0])


def is_pose_dir(d):
    for f in glob.glob(os.path.join(d, "*.json"))[:5]:
        try:
            _ts(f)
            return True
        except ValueError:
            pass
    return False


def load_dir(d):
    fs = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            fs.append((_ts(f), f))
        except ValueError:
            continue
    fs.sort()
    t = np.array([x for x, _ in fs], np.float64)
    pos = np.empty((len(fs), 3), np.float64)
    rpy = np.empty((len(fs), 3), np.float64)
    for i, (_, f) in enumerate(fs):
        j = json.load(open(f))
        pos[i] = [j["x"], j["y"], j["z"]]
        rpy[i] = [j["roll"], j["pitch"], j["yaw"]]
    return t, pos, rpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="单个 pose 目录,或含多个的根目录")
    ap.add_argument("--out", required=True, help="输出 hdf5")
    a = ap.parse_args()
    if is_pose_dir(a.src):
        dirs, rel_base = [a.src], os.path.dirname(a.src.rstrip("/"))
    else:
        dirs = sorted(dp for dp, _, _ in os.walk(a.src) if is_pose_dir(dp))
        rel_base = a.src
    if not dirs:
        raise SystemExit(f"{a.src} 下没找到 pose 目录")
    with h5py.File(a.out, "w") as h:
        for d in dirs:
            t, pos, rpy = load_dir(d)
            name = os.path.relpath(d, rel_base)
            g = h.create_group(name)
            g.create_dataset("t", data=t)
            g.create_dataset("pos", data=pos)
            g.create_dataset("rpy", data=rpy)
            print(f"  {name}: {len(t)} 帧")
    print(f"已打包 {len(dirs)} 个目录 -> {a.out}")


if __name__ == "__main__":
    main()

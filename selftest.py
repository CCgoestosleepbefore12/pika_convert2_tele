#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""numpy 自测:合成已知 ᴮʳT_P0,造标定点 + 跨 session 轨迹,验证标定复原 + 转换复原(精确)。

需 numpy/scipy(docker 里跑):python selftest.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import make_T, inv_T, T_to_pose_dict  # noqa: E402
from calib_tip_to_base import calibrate                # noqa: E402
from convert_tip_to_base import convert                # noqa: E402


def _rng_T(rng):
    R = Rot.random(random_state=rng).as_matrix()
    t = rng.normal(size=3)
    return make_T(R, t)


def _euler_pose(T):
    rpy = Rot.from_matrix(T[:3, :3]).as_euler("xyz")
    return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
            "roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])}


def run():
    rng = np.random.default_rng(0)
    T_Br_P0_true = _rng_T(rng)
    d = 0.074
    corners = np.array([[.05, .02, 0], [.18, .03, .01], [.04, .16, .09], [.15, .12, .05], [.10, .07, .13]])

    # 标定 session:pika 基站任意漂
    T_Bp_P0_c = _rng_T(rng)

    def pika_pts(idx):
        pts = []
        for k in idx:
            T_P0_Pj = make_T(Rot.random(random_state=rng).as_matrix(), corners[k])
            pts.append(_euler_pose(T_Bp_P0_c @ T_P0_Pj))
        return {"P0": _euler_pose(T_Bp_P0_c), "points": pts}

    def robot_pts(idx):
        pts = []
        T_F_G = make_T(np.eye(3), [0, 0, d])
        for k in idx:
            q = T_Br_P0_true @ np.r_[corners[k], 1.0]            # 角点在 robot base
            R_f = Rot.random(random_state=rng).as_matrix()
            t_f = q[:3] - R_f @ np.array([0, 0, d])              # 法兰位置使 法兰·ᶠT_G = 角点
            T_f = make_T(R_f, t_f)
            assert np.allclose((T_f @ T_F_G)[:3, 3], q[:3])
            quat = Rot.from_matrix(R_f).as_quat()
            pts.append({"x": t_f[0], "y": t_f[1], "z": t_f[2],
                        "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3]})
        return {"d": d, "points": pts}

    cal, val = [0, 1, 2, 3], [4]
    T_Br_P0, rep = calibrate(pika_pts(cal), robot_pts(cal), pika_pts(val), robot_pts(val))
    err_t = np.abs(T_Br_P0[:3, 3] - T_Br_P0_true[:3, 3]).max() * 1000
    err_R = np.degrees(np.linalg.norm(Rot.from_matrix(T_Br_P0[:3, :3].T @ T_Br_P0_true[:3, :3]).as_rotvec()))
    print(f"[1] 标定复原: 平移 {err_t:.2e}mm  旋转 {err_R:.2e}°  残差 mean {rep['mean_mm']:.2e}mm  验证 {rep['val_mean_mm']:.2e}mm")

    # 转换 session:换一个漂移;首帧=工装(P0=单位阵,符合"每条从工装起步")
    T_Bp_P0_t = _rng_T(rng)
    traj_P0 = [np.eye(4)] + [make_T(Rot.random(random_state=rng).as_matrix(), rng.normal(size=3) * 0.05)
                             for _ in range(11)]
    frames = [_euler_pose(T_Bp_P0_t @ Tp) for Tp in traj_P0]
    truth = [T_Br_P0_true @ Tp for Tp in traj_P0]            # robot base 真值(首帧=真值锚)
    got = convert(frames, T_Br_P0)
    pos_err = max(np.linalg.norm(g[:3, 3] - t[:3, 3]) for g, t in zip(got, truth)) * 1000
    anchor_err = np.linalg.norm(got[0][:3, 3] - truth[0][:3, 3]) * 1000   # 首帧(工装)应落在真值锚上
    print(f"[2] 跨 session 转换: 位置最大误差 {pos_err:.2e}mm  首帧落真值锚 {anchor_err:.2e}mm")

    ok = err_t < 1e-6 and err_R < 1e-4 and pos_err < 1e-6 and rep["val_mean_mm"] < 1e-6
    print("\n[selftest]", "通过 ✓" if ok else "失败 ✗")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

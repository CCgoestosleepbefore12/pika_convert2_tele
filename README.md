# tip2base —— Pika 尖端 → 松灵 Robot Base(放置版)

把 Pika 手持夹爪的尖端轨迹搬进松灵 robot base 坐标系。**只搬位置**(朝向保留 pika 约定、**不做 Y**),用于"看位置 / 可达性",标定精度 **~7mm**。

> 取代旧的 rebase+Y+anchor 方案(`../convert_umi_eef.py` / `../calib_Y.json`,已 superseded)。

## 核心公式
```
ᴮʳT_Pi = ᴮʳT_P0 · (ᴮᵖT_P0)⁻¹ · ᴮᵖT_Pi
         └ 标定常量    └ 本条轨迹自己的第一帧(消基站漂移)
```
- `ᴮʳT_P0`:pika 第一帧(固定工装)→ robot base,由**一条录制的 N 个角点**标定。
- pika raw = **尖端**位姿;robot/tele = `eef_quaternion`(法兰),**d=13.8cm** 沿法兰 +z 到尖端。

## 硬性前提
**每条录制(标定 + 所有要转换的 episode)都从同一固定工装起步**,frame0 = P0。这是跨 session 复用 `ᴮʳT_P0` 的根基。

## 模块
| 文件 | 职责 |
|---|---|
| `pose_utils.py` | 欧拉/四元数 ↔ 4×4、刚体逆、目录加载、原子写 |
| `umeyama.py` | 刚体配准 Umeyama(SVD)+ `correspond()`(最近邻指派找对应)|
| `pack_poses.py` | pose json 树 → 单个 hdf5(一次性,加速反复读)|
| `dwell_points.py` | 稳健提停留点(位置稳定+时长)→ 标定点 JSON |
| `calib_tip_to_base.py` | **核心函数**:rebase、TCP 偏移、`calibrate()`(cali/verify 法)|
| `calibrate_arm.py` | **流程入口**:点 JSON → 标定 + LOO + 丢离群 + 出图 |
| `convert_tip_to_base.py` | pika 轨迹 → robot base(单臂 pose 目录/json,或 **synced hdf5 双臂**)|
| `selftest.py` | 合成数据自测(精确复原)|

## 采集协议(每臂一条连续录制)⭐
1. 夹爪放**固定工装**,停 **5s**(= 起点 P0)。
2. 依次碰 **6–8 个点,每个停满 5s**,**铺开、立体**:x/y/z 都拉开,**别两点同位、别全共面**。
3. 全程**基站(lighthouse)别动**(两条录制之间会漂 ~25mm)。
4. 松灵端**碰同样的点**(顺序可不同,标定自动配对)。
- 停得不够久/碰歪都会被工具发现并处理(见下)。

## 用法
```bash
FD=.../tip2base
# 0) 自测(docker,需 numpy/scipy)
python $FD/selftest.py

# 1) pika:打包(一次)→ 提 N 角点(从工装起步 → --skip_first)
python $FD/pack_poses.py --src <pika pose 目录或树> --out pika_packed.hdf5
python $FD/dwell_points.py --pika pika_packed.hdf5 --group <组名> --n 8 --skip_first --out pika_X.json

# 2) 遥操:提 N 角点
#    ⚠ 看第一段在不在 frame0:在=机械臂 home,要 --skip_first;不在=全是角点,不 skip
python $FD/dwell_points.py --tele <episode.hdf5> --arm X --d 0.138 --n 8 [--skip_first] --out robot_X.json

# 3) 标定 + LOO + 出图(某角点碰歪就 --drop_worst 丢掉一个)
python $FD/calibrate_arm.py --pika pika_X.json --robot robot_X.json --arm X \
    --drop_worst --out_T T_Br_P0_X.json --out_png X_calib.png

# 4) 转换 pika episode → robot base
# 4a) synced hdf5(sync_umi_raw 产,observations/pose_left|right) —— 双臂一条命令
python $FD/convert_tip_to_base.py --traj <episode.hdf5> \
    --calib_l T_Br_P0_l.json --calib_r T_Br_P0_r.json --out ep_robotbase.json
#   输出: {"left":[...],"right":[...]}(robot base)+ .npy (T,14)=[左7,右7]
# 4b) 单臂 pose 目录/json
python $FD/convert_tip_to_base.py --traj <pika_X 目录> --calib T_Br_P0_X.json --out ep_X.json
#   输出: {"frames":[...]} + .npy (T,7)=[x,y,z,qx,qy,qz,qw]
```
> 输入 hdf5 的 `pose_left|right` 为 (T,6) 欧拉(基站系)。位置精度 ~7mm,朝向 pika 约定(未做 Y)。
> ⚠ episode 必须从同一固定工装起步(frame0=P0);首帧会精确落在该臂的锚 `ᴮʳT_P0` 上(代数必然,可用来自检)。
> 还没做:夹爪 / 三相机 / 20 维 dexechain 混训格式(要混训再补,并补 Y)。

## 调参排坑(实战经验)
- **角点没停满 5s** → 被 `--min_dur`(pika 默认 3.5s)滤掉 → 降 `--min_dur`(如 2.0)找回。
- **手持抖动大** → 停留判定半径 `--R`(默认 0.04m)调大。
- **某角点碰歪(残差离群)** → `--drop_worst` 自动丢残差最大的一个点。
- **skip 与否**:pika 起点常握住(skip);遥操看第一段帧号(在 0 附近=home 要 skip)。

## 精度判读:看 LOO,不是"拟合"
- **拟合残差**:全部点参与标定再考自己 → **偏乐观**(点少会过拟合)。
- **LOO(留一交叉验证)**:逐点用其余点标定再预测自己 → **真实泛化精度**。
- 拟合 ≈ LOO 且都小 = 好;LOO ≫ 拟合 = 过拟合(点没铺开)。
- 实测两臂:**拟合 ~5mm / LOO ~7mm**(d=13.8cm、8 点铺开、单条录制)。

## 已知边界
- **只搬位置**:输出朝向是 pika 轴约定(x前 z上),与松灵差固定旋转 Y(~106°)。要混训需末尾右乘 Y(本版不做)。
- 全局精度 ~7mm,瓶颈在 pika 基站一致性(单条录制内最好;跨录制会多漂 ~25mm,故标定走单条 + LOO)。
- pika raw 必须确为尖端;d 必须实测(本套 13.8cm)。

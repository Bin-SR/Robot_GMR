import pickle
import numpy as np
import pandas as pd


pkl_file = "/home/si/GMR/motions/G1/pkl/Leo.pkl"
csv_file = "/home/si/GMR/motions/G1/csv/Leo.csv"


# 读取 GMR 的 pickle 文件
with open(pkl_file, "rb") as f:
    motion_data = pickle.load(f)


# 获取数据
root_pos = motion_data["root_pos"]
root_rot = motion_data["root_rot"]
dof_pos = motion_data["dof_pos"]


# 拼接成一个二维数组
data = np.concatenate(
    [
        root_pos,
        root_rot,
        dof_pos
    ],
    axis=1
)


# 创建列名
columns = [
    "root_x",
    "root_y",
    "root_z",

    "root_qx",
    "root_qy",
    "root_qz",
    "root_qw",
]


# G1关节名称
joint_names = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",

    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",

    "waist_yaw",
    "waist_roll",
    "waist_pitch",

    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",

    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",

    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",

    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


columns += joint_names


# 检查维度
print("data shape:", data.shape)
print("number of columns:", len(columns))


# 如果列数不一致，直接报错
if data.shape[1] != len(columns):
    raise ValueError(
        f"数据列数 {data.shape[1]} 与列名数量 {len(columns)} 不一致"
    )


# 保存真正的 CSV
df = pd.DataFrame(data, columns=columns)

df.to_csv(
    csv_file,
    index=False
)


print(f"CSV saved to: {csv_file}")

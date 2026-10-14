"""X1 flat-terrain motion imitation configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.assets.robots.x1.x1_constants import X1_ACTION_SCALE, get_x1_robot_cfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


X1_TRACKING_BODIES = (
  "x1-body",
  "body_yaw",
  "body_roll",
  "body_pitch",
  "left_shoulder_pitch",
  "left_shoulder_roll",
  "left_shoulder_yaw",
  "left_elbow_pitch",
  "left_elbow_yaw",
  "left_wrist_pitch",
  "left_wrist_roll",
  "right_shoulder_pitch",
  "right_shoulder_roll",
  "right_shoulder_yaw",
  "right_elbow_pitch",
  "right_elbow_yaw",
  "right_wrist_pitch",
  "right_wrist_roll",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_pitch_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_pitch_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
)


def x1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_x1_robot_cfg()}

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="x1-body", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="x1-body", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = X1_ACTION_SCALE

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.anchor_body_name = "x1-body"
  motion.body_names = X1_TRACKING_BODIES

  # X1 uses these names in its MJCF sensor block, not G1's names.
  cfg.observations["actor"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/body-linear-vel"
  )
  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/body-angular-velocity"
  )
  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/body-linear-vel"
  )
  cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/body-angular-velocity"
  )

  cfg.events["base_com"].params["asset_cfg"].body_names = ("x1-body",)
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll",
    "right_wrist_roll",
  )
  cfg.viewer.body_name = "x1-body"

  if not has_state_estimation:
    terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=terms, concatenate_terms=True, enable_corruption=True
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"

  return cfg

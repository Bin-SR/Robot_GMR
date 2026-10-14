"""X1 robot configuration for mjlab motion imitation."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

X1_XML: Path = SRC_PATH / "assets" / "robots" / "x1" / "xyber_x1_serial.xml"
assert X1_XML.exists(), f"X1 MJCF not found: {X1_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X1_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Initial values. Replace these with identified X1 motor/gearbox parameters.
X1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("body_yaw", "body_roll", "body_pitch"),
  stiffness=100.0,
  damping=5.0,
  effort_limit=50.0,
  armature=0.02,
)
X1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*shoulder.*", ".*elbow.*"),
  stiffness=80.0,
  damping=4.0,
  effort_limit=25.0,
  armature=0.01,
)
X1_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*wrist.*",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=5.0,
  armature=0.005,
)
X1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_pitch",),
  stiffness=150.0,
  damping=8.0,
  effort_limit=150.0,
  armature=0.05,
)
X1_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_roll", ".*hip_yaw"),
  stiffness=100.0,
  damping=5.0,
  effort_limit=50.0,
  armature=0.03,
)
X1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*knee_pitch",),
  stiffness=150.0,
  damping=8.0,
  effort_limit=150.0,
  armature=0.05,
)
X1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*ankle_pitch", ".*ankle_roll"),
  stiffness=50.0,
  damping=3.0,
  effort_limit=18.0,
  armature=0.01,
)

X1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X1_ACTUATOR_WAIST,
    X1_ACTUATOR_ARM,
    X1_ACTUATOR_WRIST,
    X1_ACTUATOR_HIP_PITCH,
    X1_ACTUATOR_HIP,
    X1_ACTUATOR_KNEE,
    X1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

X1_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8),
  joint_pos={
    "body_yaw": 0.0,
    "body_roll": 0.0,
    "body_pitch": 0.0,
    ".*hip_pitch": -0.15,
    ".*hip_roll": 0.0,
    ".*hip_yaw": 0.0,
    ".*knee_pitch": 0.3,
    ".*ankle_pitch": -0.15,
    ".*ankle_roll": 0.0,
    ".*shoulder_pitch": 0.2,
    ".*shoulder_roll": 0.0,
    ".*shoulder_yaw": 0.0,
    ".*elbow_pitch": 0.5,
    ".*elbow_yaw": 0.0,
    ".*wrist_pitch": 0.0,
    ".*wrist_roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

X1_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim=1,
  contype=1,
  conaffinity=1,
)


def get_x1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=X1_HOME,
    collisions=(X1_COLLISION,),
    spec_fn=get_spec,
    articulation=X1_ARTICULATION,
  )


# Used by the action config. Regex keys are accepted by mjlab action matching.
X1_ACTION_SCALE = {
  "body_yaw": 0.20,
  "body_roll": 0.20,
  "body_pitch": 0.20,
  ".*shoulder.*": 0.25,
  ".*elbow.*": 0.25,
  ".*wrist.*": 0.15,
  ".*hip_pitch": 0.25,
  ".*hip_roll": 0.25,
  ".*hip_yaw": 0.25,
  ".*knee_pitch": 0.20,
  ".*ankle_pitch": 0.20,
  ".*ankle_roll": 0.20,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  viewer.launch(Entity(get_x1_robot_cfg()).spec.compile())

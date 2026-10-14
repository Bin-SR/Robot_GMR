from mjlab.tasks.registry import register_mjlab_task

from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import x1_flat_tracking_env_cfg
from .rl_cfg import x1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="X1-Tracking",
  env_cfg=x1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=x1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=x1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

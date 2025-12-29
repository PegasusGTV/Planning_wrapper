# Planning Wrapper

A Python package that provides planning-friendly wrappers and utilities for [ManiSkill3](https://github.com/haosulab/ManiSkill) environments. This package enables efficient state cloning/restoration, planning observation extraction, and task-specific adapters for manipulation planning algorithms.

## Features

- **State Management**: Clone and restore environment states for planning algorithms
- **Planning Observations**: Extract structured observations optimized for planning
- **Task Adapters**: Task-specific adapters for saving/restoring task state (e.g., goal positions)
- **Robot Control Utilities**: Easy access to joint positions, velocities, and controller information
- **Compatible with ManiSkill3**: Works seamlessly with ManiSkill environments

## Installation

### Prerequisites

- Python >= 3.9
- [ManiSkill3](https://github.com/haosulab/ManiSkill) installed

### Install from Source

```bash
git clone https://github.com/PegasusGTV/Planning_wrapper.git
cd Planning_wrapper
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
import mani_skill.envs  # Register ManiSkill environments

from planning_wrapper import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter

# Create environment
env = gym.make(
    "PushT-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)

# Wrap with planning wrapper
adapter = PushTTaskAdapter()
wrapper = ManiSkillPlanningWrapper(env, adapter=adapter)

# Reset environment
obs, info = wrapper.reset(seed=0)

# Clone state for planning
snapshot = wrapper.clone_state()

# ... do planning ...

# Restore state
wrapper.restore_state(snapshot)

# Get planning-friendly observations
planning_obs = wrapper.get_planning_obs(obs)
```

## Usage Examples

### Basic State Cloning

```python
from planning_wrapper import ManiSkillPlanningWrapper
from planning_wrapper.adapters import PushTTaskAdapter

env = gym.make("PushT-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pose")
wrapper = ManiSkillPlanningWrapper(env, adapter=PushTTaskAdapter())

obs, info = wrapper.reset(seed=0)

# Clone current state
snapshot = wrapper.clone_state()

# Take some actions
for _ in range(10):
    action = wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)

# Restore to original state
wrapper.restore_state(snapshot)
```

### Planning Observations

```python
# Get structured planning observations
planning_obs = wrapper.get_planning_obs(obs)

# Access individual components
qpos = planning_obs["qpos"]        # Joint positions
qvel = planning_obs["qvel"]       # Joint velocities
tcp_pose = planning_obs["tcp_pose"]  # End-effector pose
goal_pos = planning_obs["goal_pos"]   # Goal position
obj_pose = planning_obs["obj_pose"]  # Object pose

# Flatten for algorithms that need 1D vectors
flat_obs = wrapper.flatten_planning_obs(planning_obs)
```

### Robot Control Utilities

```python
# Get full robot state
qpos = wrapper.get_qpos()      # All joint positions
qvel = wrapper.get_qvel()      # All joint velocities

# Get controlled joints only
controlled_indices = wrapper.controlled_joint_indices()
controlled_qpos = wrapper.controlled_qpos()
controlled_qvel = wrapper.controlled_qvel()

# Print controller summary
wrapper.print_controller_summary()
```

## API Reference

### `ManiSkillPlanningWrapper`

Main wrapper class for ManiSkill environments.

#### Methods

- **`clone_state() -> Dict[str, Any]`**: Create a snapshot of the current environment state
- **`restore_state(snapshot: Dict[str, Any]) -> None`**: Restore environment to a previous state
- **`get_planning_obs(obs: Dict) -> Dict[str, np.ndarray]`**: Extract planning-friendly observations
- **`flatten_planning_obs(planning_obs: Dict) -> np.ndarray`**: Flatten observations to 1D array
- **`get_qpos() -> np.ndarray`**: Get all joint positions
- **`get_qvel() -> np.ndarray`**: Get all joint velocities
- **`controlled_joint_indices() -> np.ndarray`**: Get indices of controlled joints
- **`controlled_qpos() -> np.ndarray`**: Get positions of controlled joints only
- **`controlled_qvel() -> np.ndarray`**: Get velocities of controlled joints only
- **`print_controller_summary() -> None`**: Print information about the controller

### Task Adapters

Task adapters handle task-specific state (e.g., goal positions) that need to be saved/restored.

#### `PushTTaskAdapter`

Adapter for the PushT task that saves/restores:
- Goal tee pose
- End-effector goal position
- Goal rotation and offset parameters

#### Creating Custom Adapters

```python
from planning_wrapper.adapters import BaseTaskAdapter

class MyTaskAdapter(BaseTaskAdapter):
    def get_task_state(self, env: Any) -> Dict[str, Any]:
        # Extract task-specific state
        return {"goal": env.goal_position}
    
    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:
        # Restore task-specific state
        env.goal_position = task_state["goal"]
```

## Project Structure

```
Planning_wrapper/
├── planning_wrapper/
│   ├── __init__.py
│   ├── wrappers/
│   │   └── maniskill_planning.py    # Main wrapper class
│   ├── adapters/
│   │   ├── base.py                  # Base adapter interface
│   │   └── pusht.py                 # PushT task adapter
│   └── utils/
├── examples/
│   ├── inspect_pusht.py            # Inspect PushT environment
│   ├── pusht_demo1.py               # Simple planning demo
│   ├── pusht_demo2.py               # Longer planning demo
│   └── replay_smoke.py              # State cloning test
├── tests/
│   └── test_replay.py
├── pyproject.toml
└── README.md
```

## Examples

See the `examples/` directory for complete working examples:

- **`inspect_pusht.py`**: Inspect the PushT environment structure and observations
- **`pusht_demo1.py`**: Simple planning demo with basic policy
- **`pusht_demo2.py`**: Longer planning demo with rendering
- **`replay_smoke.py`**: Test state cloning and restoration

Run examples:
```bash
python -m examples.inspect_pusht
python -m examples.pusht_demo1
python -m examples.pusht_demo2
```

## Requirements

- `numpy`
- `gymnasium` (or `gym`)
- `mani_skill` (ManiSkill3)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Acknowledgments

Built for use with [ManiSkill3](https://github.com/haosulab/ManiSkill), a unified benchmark for generalizable manipulation skills.

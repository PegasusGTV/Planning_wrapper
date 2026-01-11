# Planning Wrapper

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="videos/pusht_example2.gif" alt="PushT Planning Demo" width="600"/>
  <p><em>PushT Task: Robot manipulation planning with state cloning and restoration</em></p>
</div>

A Python package that provides planning-friendly wrappers and utilities for [ManiSkill3](https://github.com/haosulab/ManiSkill) environments. This package enables efficient state cloning/restoration, planning observation extraction, and task-specific adapters for manipulation planning algorithms.

## 🎯 Overview

Planning Wrapper bridges the gap between ManiSkill3 environments and planning algorithms by providing:

- **Efficient State Management**: Clone and restore complete environment states for planning algorithms
- **Structured Observations**: Extract planning-friendly observations with consistent structure
- **Task-Specific Adapters**: Handle task-specific state (e.g., goal positions) that need to be saved/restored
- **Robot Control Utilities**: Easy access to joint positions, velocities, and controller information
- **Seamless Integration**: Works transparently with ManiSkill3 environments

## ✨ Features

- **State Cloning & Restoration**: Create snapshots of environment state and restore them efficiently
- **Planning Observations**: Extract structured observations optimized for planning algorithms
- **Task Adapters**: Task-specific adapters for saving/restoring task state (e.g., goal positions, object states)
- **Robot Control Utilities**: Access joint positions, velocities, and controller information
- **Multi-Object Support**: Handle environments with multiple objects (e.g., shelf retrieval)
- **Observation Filtering**: Optional filtering of object orientations for planning
- **Compatible with ManiSkill3**: Works seamlessly with all ManiSkill environments

## 📦 Installation

### Prerequisites

- Python >= 3.9 (3.11 recommended)
- Git (for installing ManiSkill3)
- [ManiSkill3](https://github.com/haosulab/ManiSkill) installed and configured

### Linux Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/PegasusGTV/Planning_wrapper.git
cd Planning_wrapper
```

#### Step 2: Create a Virtual Environment (Recommended)

Using conda (recommended):
```bash
conda create -n planning_wrapper python=3.11 -y
conda activate planning_wrapper
```

Or using venv:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install System Dependencies (Linux)

Install Pinocchio via conda (recommended):
```bash
conda install -c conda-forge pinocchio -y
```

Or install system packages if using system Python:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libpinocchio-dev

# Fedora/CentOS
sudo dnf install -y pinocchio-devel
```

#### Step 4: Install Python Dependencies

Upgrade pip first:
```bash
pip install --upgrade pip
```

Install project dependencies:
```bash
pip install -r requirements.txt
```

#### Step 5: Install ManiSkill3

```bash
pip install git+https://github.com/haosulab/ManiSkill.git
```

This may take several minutes as it downloads and installs many dependencies.

#### Step 6: Install PyTorch

Install PyTorch based on your CUDA version (if you have a GPU):

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

#### Step 7: Install Planning Wrapper

```bash
pip install -e .
```

#### Step 8: Verify Installation

Test that everything is installed correctly:
```bash
python -c "import planning_wrapper; print('✅ planning_wrapper installed!')"
python -c "import mani_skill; print('✅ ManiSkill3 installed!')"
python -c "import pinocchio; print('✅ pinocchio installed!')"
python -c "import torch; print('✅ PyTorch version:', torch.__version__)"
```

### Windows Installation

For Windows users, we provide a detailed step-by-step guide. Please follow the instructions in **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** for complete Windows installation instructions.

**Quick Windows Setup Summary:**

1. Install Python 3.9+ and Miniconda
2. Install Git and Visual C++ Redistributables
3. Create conda environment: `conda create -n planning_wrapper python=3.11 -y`
4. Install dependencies: `pip install -r requirements.txt`
5. Install ManiSkill3: `pip install git+https://github.com/haosulab/ManiSkill.git`
6. Install Pinocchio: `conda install -c conda-forge pinocchio -y`
7. Install PyTorch (CPU version recommended for Windows): `pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu`
8. Install Planning Wrapper: `pip install -e .`

For detailed Windows instructions, troubleshooting, and platform-specific notes, see **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)**.

### Install Dependencies

The package requires:
- `numpy>=1.22.0,<2.0.0` (NumPy 2.x is not yet fully supported)
- `gymnasium>=0.29.0`
- `mani_skill` (ManiSkill3)
- `pinocchio` (for kinematics, install via conda on both Linux and Windows)

See `requirements.txt` for a complete list of dependencies.

## 🚀 Quick Start

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

## 📖 Usage Examples

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
tcp_pose = planning_obs["tcp_pose"]  # End-effector pose (7D: x, y, z, qx, qy, qz, qw)
goal_pos = planning_obs["goal_pos"]   # Goal position (3D)
obj_pose = planning_obs["obj_pose"]  # Object pose (7D)

# For multi-object environments (e.g., shelf retrieval)
if "target_obj_pose" in planning_obs:
    target_obj_pose = planning_obs["target_obj_pose"]
if "obj_poses" in planning_obs:
    all_obj_poses = planning_obs["obj_poses"]

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

# Get controller action bounds
low, high = wrapper.get_controller_bounds()

# Print controller summary
wrapper.print_controller_summary()
```

### Shelf Retrieval Environment

```python
import envs  # Register shelf retrieval environment
from planning_wrapper.adapters import ShelfRetrieveTaskAdapter

env = gym.make(
    "ObjectRetrieveFromShelf-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pose",
)

adapter = ShelfRetrieveTaskAdapter()
wrapper = ManiSkillPlanningWrapper(env, adapter=adapter)

obs, info = wrapper.reset(seed=0)
planning_obs = wrapper.get_planning_obs(obs)

# Access multi-object information
target_obj_id = planning_obs.get("target_obj_id")
target_obj_pose = planning_obs.get("target_obj_pose")
```

### Observation Filtering

```python
# Hide object orientation in observations (useful for some planning algorithms)
wrapper = ManiSkillPlanningWrapper(
    env, 
    adapter=adapter,
    hide_obj_orientation=True  # Sets object quaternion to [0, 0, 0, 1]
)
```

## 📚 API Reference

### `ManiSkillPlanningWrapper`

Main wrapper class for ManiSkill environments.

#### Constructor

```python
ManiSkillPlanningWrapper(
    env: gym.Env,
    adapter: Optional[BaseTaskAdapter] = None,
    hide_obj_orientation: bool = False
)
```

**Parameters:**
- `env`: ManiSkill environment instance
- `adapter`: Optional task adapter for saving/restoring task-specific state
- `hide_obj_orientation`: If `True`, filters out object orientation from observations

#### Methods

##### State Management

- **`clone_state() -> Dict[str, Any]`**: Create a snapshot of the current environment state. Returns a dictionary containing:
  - `sim_state`: Simulation state (actors, articulations)
  - `controller_state`: Controller state
  - `task_state`: Task-specific state (if adapter is provided)

- **`restore_state(snapshot: Dict[str, Any]) -> None`**: Restore environment to a previous state snapshot.

##### Observations

- **`get_planning_obs(obs: Dict) -> Dict[str, np.ndarray]`**: Extract planning-friendly observations from a state_dict observation. Returns a dictionary with:
  - `qpos`: Joint positions
  - `qvel`: Joint velocities
  - `tcp_pose`: End-effector pose (7D: position + quaternion)
  - `goal_pos`: Goal position (3D)
  - `obj_pose`: Object pose (7D)
  - `target_obj_pose`: Target object pose (for multi-object tasks, optional)
  - `obj_poses`: All object poses (for multi-object tasks, optional)
  - `target_obj_id`: Target object ID (for multi-object tasks, optional)

- **`flatten_planning_obs(planning_obs: Dict) -> np.ndarray`**: Flatten observations to a 1D numpy array for algorithms that require vector inputs.

##### Robot State

- **`get_qpos() -> np.ndarray`**: Get all joint positions of the robot.
- **`get_qvel() -> np.ndarray`**: Get all joint velocities of the robot.
- **`controlled_joint_indices() -> np.ndarray`**: Get indices of controlled joints.
- **`controlled_qpos() -> np.ndarray`**: Get positions of controlled joints only.
- **`controlled_qvel() -> np.ndarray`**: Get velocities of controlled joints only.
- **`get_controller_bounds() -> tuple`**: Get action bounds for the controller (returns `(low, high)` tuple).

##### Utilities

- **`print_controller_summary() -> None`**: Print detailed information about the controller and robot configuration.

##### Environment Interface

The wrapper implements the standard Gymnasium interface:
- **`reset(seed=None, options=None) -> tuple`**: Reset the environment and return filtered observations.
- **`step(action) -> tuple`**: Step the environment and return filtered observations.
- **`close() -> None`**: Close the environment.

### Task Adapters

Task adapters handle task-specific state (e.g., goal positions) that need to be saved/restored during planning.

#### `BaseTaskAdapter`

Abstract base class for task adapters. Subclasses must implement:

- **`get_task_state(env) -> Dict[str, Any]`**: Extract task-specific state from the environment.
- **`set_task_state(env, task_state: Dict[str, Any]) -> None`**: Restore task-specific state to the environment.

#### `PushTTaskAdapter`

Adapter for the PushT task that saves/restores:
- `goal_tee_pose`: Goal tee pose
- `ee_goal_pose`: End-effector goal position
- `goal_rotation`: Goal rotation parameters
- `goal_offset`: Goal offset parameters

#### `ShelfRetrieveTaskAdapter`

Adapter for the ObjectRetrieveFromShelf task that saves/restores:
- `target_obj_id`: ID of the target object to retrieve
- `target_obj_pose`: Pose of the target object
- Additional task-specific state for shelf retrieval

#### Creating Custom Adapters

```python
from planning_wrapper.adapters import BaseTaskAdapter

class MyTaskAdapter(BaseTaskAdapter):
    def get_task_state(self, env: Any) -> Dict[str, Any]:
        # Extract task-specific state
        return {
            "goal": env.goal_position,
            "target_id": env.target_object_id,
        }
    
    def set_task_state(self, env: Any, task_state: Dict[str, Any]) -> None:
        # Restore task-specific state
        env.goal_position = task_state["goal"]
        env.target_object_id = task_state["target_id"]
```

## 🗂️ Project Structure

```
Planning_wrapper/
├── planning_wrapper/
│   ├── __init__.py
│   ├── wrappers/
│   │   ├── __init__.py
│   │   ├── maniskill_planning.py    # Main wrapper class
│   │   └── factory.py               # Factory functions for common setups
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base adapter interface
│   │   ├── pusht.py                 # PushT task adapter
│   │   └── shelf_retrieve.py        # Shelf retrieval task adapter
│   └── utils/
│       └── __init__.py
├── envs/
│   ├── __init__.py
│   ├── shelf_retrieve_v1.py         # Custom shelf retrieval environment
│   └── shelf_scene_builder.py       # Scene builder for shelf environment
├── examples/
│   ├── inspect_pusht.py            # Inspect PushT environment structure
│   ├── inspect_self.py              # Inspect shelf environment structure
│   ├── pusht_demo1.py               # Simple PushT planning demo
│   ├── pusht_demo2.py               # PushT demo with rendering
│   ├── pusht_demo3.py               # Advanced PushT demo
│   ├── pusht_demo4.py               # PushT demo with state cloning
│   ├── shelf_demo1.py               # Simple shelf retrieval demo
│   ├── shelf_demo2.py               # Shelf retrieval with rendering
│   ├── shelf_demo3.py               # Advanced shelf retrieval demo
│   ├── replay_smoke.py              # State cloning test for PushT
│   ├── replay_shelf_smoke.py        # State cloning test for shelf
│   └── smoke_test.py                # General smoke test
├── tests/
│   ├── test_pusht.py                # Tests for PushT wrapper
│   ├── test_replay.py               # Tests for state cloning
│   └── test_replay_shelf.py         # Tests for shelf state cloning
├── videos/                          # Demo videos and GIFs
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Python dependencies for Linux setup
├── WINDOWS_SETUP.md                 # Detailed Windows installation guide
└── README.md                        # This file
```

## 🎬 Visualizations

### PushT Task Demonstrations

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="videos/pusht_example2.gif" alt="PushT Demo 2" width="400"/>
        <br><strong>PushT Demo 2</strong>
      </td>
      <td align="center">
        <img src="videos/pusht_example3.gif" alt="PushT Demo 3" width="400"/>
        <br><strong>PushT Demo 3</strong>
      </td>
    </tr>
  </table>
</div>

### Shelf Retrieval Task Demonstrations

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="videos/object_retrival_example2.gif" alt="Shelf Retrieval Demo 2" width="400"/>
        <br><strong>Shelf Retrieval Demo 2</strong>
      </td>
      <td align="center">
        <img src="videos/object_retrieval3.gif" alt="Shelf Retrieval Demo 3" width="400"/>
        <br><strong>Shelf Retrieval Demo 3</strong>
      </td>
    </tr>
  </table>
</div>

## 🎬 Examples

The `examples/` directory contains complete working examples:

### PushT Examples

- **`inspect_pusht.py`**: Inspect the PushT environment structure and observations
- **`pusht_demo1.py`**: Simple planning demo with basic policy
- **`pusht_demo2.py`**: Planning demo with rendering
- **`pusht_demo3.py`**: Advanced planning demo with state cloning
- **`pusht_demo4.py`**: Demo with custom planning algorithm

### Shelf Retrieval Examples

- **`inspect_self.py`**: Inspect the shelf retrieval environment structure
- **`shelf_demo1.py`**: Simple shelf retrieval demo
- **`shelf_demo2.py`**: Shelf retrieval with rendering
- **`shelf_demo3.py`**: Advanced shelf retrieval with multi-object planning

### Testing Examples

- **`replay_smoke.py`**: Test state cloning and restoration for PushT
- **`replay_shelf_smoke.py`**: Test state cloning and restoration for shelf retrieval
- **`smoke_test.py`**: General smoke test for the wrapper

### Running Examples

```bash
# PushT examples
python -m examples.inspect_pusht
python -m examples.pusht_demo1
python -m examples.pusht_demo2
python -m examples.pusht_demo3
python -m examples.pusht_demo4

# Shelf retrieval examples
python -m examples.inspect_self
python -m examples.shelf_demo1
python -m examples.shelf_demo2
python -m examples.shelf_demo3

# Testing examples
python -m examples.replay_smoke
python -m examples.replay_shelf_smoke
python -m examples.smoke_test
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pusht.py
pytest tests/test_replay.py
pytest tests/test_replay_shelf.py
```

## 🔧 Requirements

- Python >= 3.9 (3.11 recommended)
- `numpy>=1.22.0,<2.0.0` (NumPy 2.x is not yet fully supported)
- `gymnasium>=0.29.0`
- `mani_skill` (ManiSkill3)
- `pinocchio` (for kinematics, install via conda: `conda install -c conda-forge pinocchio`)

**For Linux:** See `requirements.txt` for a complete list of dependencies that can be installed via pip.

**For Windows:** See `WINDOWS_SETUP.md` for detailed Windows-specific installation instructions.

See `pyproject.toml` for package metadata and optional development dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for use with [ManiSkill3](https://github.com/haosulab/ManiSkill), a unified benchmark for generalizable manipulation skills
- Inspired by the need for efficient state management in manipulation planning algorithms

## 📞 Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/PegasusGTV/Planning_wrapper/issues).

---

**Note**: This package requires ManiSkill3 to be properly installed and configured. Make sure you have followed the [ManiSkill3 installation guide](https://github.com/haosulab/ManiSkill) before using this wrapper.

# POLARIS

POLARIS (Policy Assignment for Robust Intent-Driven Skill Execution) is a robot manipulation framework that combines LLM-based task decomposition with an adaptive skill library. An LLM breaks long-horizon tasks into subgoals using a PDDL representation of the scene, then a **Policy Assignment** module selects between classical (MPPI) and learned (PPO) execution for each subgoal, retrying with the alternative on failure.

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/meganmlee/POLARIS.git
cd POLARIS
conda create -n polaris python=3.11 -y
conda activate polaris
conda install -c conda-forge pinocchio -y
```

### 2. Configure the LLM

Copy the config template and fill in your API credentials:

```bash
cp high_level_planner/config.json.example high_level_planner/config.json
```

Edit `high_level_planner/config.json`:

```json
{
  "LLM_BACKEND": "openai",
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_BASE_URL": "https://..."
}
```

Or for Gemini (Vertex AI):

```json
{
  "LLM_BACKEND": "gemini",
  "VERTEX_PROJECT": "your-gcp-project-id",
  "VERTEX_LOCATION": "us-central1"
}
```

> The LLM is only needed for `executor.py` and `llm_plan.py`. All skills and demos work without it.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/haosulab/ManiSkill.git
```

Install PyTorch for your CUDA version:

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch torchvision torchaudio
```

```bash
pip install -e .
```

### Verify

```bash
python -c "import mani_skill; print('ManiSkill3 OK')"
python -c "import envs; print('Custom envs OK')"
python -c "import torch; print('PyTorch', torch.__version__)"
```

## Project Structure

```
POLARIS/
├── envs/                   # Custom ManiSkill environments
├── skills/                 # Skill backends (MPPI and PPO per skill)
│   ├── reach/
│   ├── push_cube/
│   ├── push_o/
│   ├── pick/
│   └── place/
├── high_level_planner/     # LLM subgoal planning and closed-loop execution
│   ├── executor.py         # Main entry point: plan → skill → replan
│   ├── llm_plan.py         # Subgoal planner (LLM → pyperplan → BFS fallback)
│   ├── env_subgoal_runner.py
│   ├── domain_pusho.pddl
│   └── config.json.example
├── examples/               # Visual demos
├── planning_wrapper/       # State clone/restore for planning backends
└── checkpoints/            # Saved PPO checkpoints (auto-created)
```

## Skills

Each skill has a classical (MPPI) and learned (PPO) backend using the same environment:

| Skill | Classical | Learned |
|---|---|---|
| Reach | `reach_mpc.py` | `reach_ppo.py` |
| Push Cube | `push_cube_mpc.py` | `push_cube_ppo.py` |
| Pick | `pick_cube_mpc.py` | `pick_cube_ppo.py` |
| Place | `place_cube_mpc.py` | `place_cube_ppo.py` |
| Push O | `push_o_mpc.py` | `push_o_ppo.py` |

## Running

### Demos

Classical (MPPI) — no checkpoint needed:

```bash
python examples/reach_mpc_demo.py
python examples/pick_cube_mpc_demo.py
python examples/place_cube_mpc_demo.py
python examples/push_cube_mpc_demo.py
python examples/pusho_obstacles_demo.py
```

PPO — pre-trained checkpoints are included under `checkpoints/`:

```bash
python examples/reach_ppo_demo.py --checkpoint checkpoints/Reach/final_ckpt.pt
python examples/push_cube_ppo_demo.py --checkpoint checkpoints/PushCube/final_ckpt.pt
```

### Train a PPO skill

```bash
python skills/reach/reach_ppo.py
python skills/reach/reach_ppo.py --num_envs 512 --total_timesteps 10_000_000 --seed 42

# Evaluate a checkpoint
python skills/reach/reach_ppo.py --evaluate \
    --checkpoint checkpoints/Reach-WithObstacles-v1__1__<timestamp>/final_ckpt.pt
```

### Full task execution

`executor.py` is the main entry point for end-to-end planning and execution:

```bash
# Full Skill Switching
python high_level_planner/executor.py --skill auto

# MPPI (no checkpoint needed)
python high_level_planner/executor.py --skill mpc

# PPO (requires trained checkpoints; defaults to any found under checkpoints/)
python high_level_planner/executor.py

# Specific environment
python high_level_planner/executor.py --env-id "PushO-Scattered"
```

## Docker

```bash
cd docker_utils
./build.sh      # build image
./run.sh        # start container (GPU + X11)
./terminal.sh   # reconnect to running container
```

## Contributors
- Abhishek Mathur
- Megan Lee
- Tom Gao

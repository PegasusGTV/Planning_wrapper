# Docker utils for POLARIS

**Expected layout** (build/run from the directory that contains `docker_ws`):

```
your_repo_root/
├── docker_ws/
│   └── POLARIS/          # this repo (clone or copy here)
│       ├── docker_utils/ # this folder (Dockerfile + scripts)
│       └── ...
├── Dockerfile            # copy from docker_utils/Dockerfile, or symlink
├── build.sh
├── run.sh
└── terminal.sh
```

**Usage:** From that repo root (so `docker_ws/POLARIS` exists):

1. **Build:** `./build.sh` — builds image `polaris:latest`.
2. **Run:** `./run.sh` — starts container, mounts current dir as `/workspace`, opens a shell. Exit leaves the container running.
3. **Reconnect:** `./terminal.sh` — opens another shell in the same container.

Requires Docker, NVIDIA Container Toolkit, and X11 for GUI.

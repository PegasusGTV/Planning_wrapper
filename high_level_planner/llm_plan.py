"""
PushO-with-obstacles subgoal planning.

Order: LLM proposes logical steps → optional repair against a full symbolic plan
(pyperplan) so the sequence is sound → else full PDDL plan → else BFS.
Output: list of {"skill", "state"}.
"""
import json
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*deprecated.*")

GRID = 10
TABLE_BOUND = 0.30
TILE_SIZE_M = (2.0 * TABLE_BOUND) / GRID

NUM_OBSTACLES = 10
NUM_PICKABLE = 10

VALID_SKILLS = frozenset({"reach", "push_disk", "pick", "place"})


def _region_to_name(idx: int) -> str:
    row, col = idx // GRID, idx % GRID
    return f"r_{row}_{col}"


def _name_to_idx(name: str) -> int:
    m = re.match(r"r_(\d+)_(\d+)", name.strip())
    if not m:
        return 0
    row, col = int(m.group(1)), int(m.group(2))
    return row * GRID + col


def _xy_to_region(x: float, y: float) -> int:
    s = TABLE_BOUND
    cx = int((float(x) + s) / (2 * s) * GRID)
    cy = int((float(y) + s) / (2 * s) * GRID)
    cx = max(0, min(GRID - 1, cx))
    cy = max(0, min(GRID - 1, cy))
    return cy * GRID + cx


def region_to_xy(name: str) -> tuple[float, float]:
    """Center of a grid tile in world XY (metres). Inverse of _xy_to_region."""
    m = re.match(r"r_(\d+)_(\d+)", name.strip())
    if not m:
        return 0.0, 0.0
    row, col = int(m.group(1)), int(m.group(2))
    x = (col + 0.5) / GRID * 2 * TABLE_BOUND - TABLE_BOUND
    y = (row + 0.5) / GRID * 2 * TABLE_BOUND - TABLE_BOUND
    return float(x), float(y)


def place_fallback_candidates(region: str, n: int = 4) -> list[str]:
    """Alternative placement regions when the primary target fails.

    Walks outward from *region* in steps of 2, 3, … tiles along the four
    cardinal directions so retries are spread across the table rather than
    clustered in neighbouring cells.  The original region is excluded.
    """
    m = re.match(r"r_(\d+)_(\d+)", region.strip())
    if not m:
        return []
    row, col = int(m.group(1)), int(m.group(2))
    candidates: list[str] = []
    for dist in range(2, GRID):
        for dr, dc in [(0, dist), (0, -dist), (dist, 0), (-dist, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                name = f"r_{nr}_{nc}"
                if name not in candidates:
                    candidates.append(name)
        if len(candidates) >= n:
            break
    return candidates[:n]


def _adjacent_pairs():
    out = []
    for i in range(GRID):
        for j in range(GRID):
            r = i * GRID + j
            if j + 1 < GRID:
                out.append((r, r + 1))
            if j - 1 >= 0:
                out.append((r, r - 1))
            if i + 1 < GRID:
                out.append((r, r + GRID))
            if i - 1 >= 0:
                out.append((r, r - GRID))
    return out


def _region_layout_comment() -> str:
    return (
        f"; REGIONS: r_m_n, m row 0..{GRID - 1}, n col 0..{GRID - 1}. "
        f"Tile size ~{TILE_SIZE_M:.4f} m. Adjacent = edge neighbors."
    )


def state_to_problem(disk_xy: np.ndarray, goal_xy: np.ndarray, ee_xy: np.ndarray, obstacle_regions: list, stuck_obstacles: set = None) -> str:
    rd = _xy_to_region(disk_xy[0], disk_xy[1])
    rg = _xy_to_region(goal_xy[0], goal_xy[1])
    re = _xy_to_region(ee_xy[0], ee_xy[1])
    obstacles = list(obstacle_regions)[:NUM_OBSTACLES]
    n_obstacles = len(obstacles)
    n_pickable = min(NUM_PICKABLE, n_obstacles)
    stuck = stuck_obstacles or set()
    region_set = set(obstacles)
    region_names = [_region_to_name(i) for i in range(GRID * GRID)]
    objects = "robot1 - robot disk - disk " + " ".join(f"obstacle{i}" for i in range(n_obstacles)) + " - obstacle " + " ".join(region_names) + " - region"
    init = [
        f"(robot-at robot1 {_region_to_name(re)})",
        f"(object-at disk {_region_to_name(rd)})",
        f"(goal-at {_region_to_name(rg)})",
        "(hand-empty robot1)",
    ]
    for i in range(n_pickable):
        if i not in stuck:
            init.append(f"(pickable obstacle{i})")
        # stuck obstacles get no pickable/push-only predicate → treated as permanent walls
    for i in range(n_pickable, n_obstacles):
        if i not in stuck:
            init.append(f"(push-only obstacle{i})")
    for i, r in enumerate(obstacles):
        init.append(f"(obstacle-at obstacle{i} {_region_to_name(r)})")
    # (clear): cell has no obstacle — used by pick/place.
    for i in range(GRID * GRID):
        if i not in region_set:
            init.append(f"(clear {_region_to_name(i)})")
    # (disk-clear): cell has a full 1-tile buffer around all obstacles — used only by push_disk.
    disk_blocked_set = _expand_cells_by_1(region_set)
    for i in range(GRID * GRID):
        if i not in disk_blocked_set:
            init.append(f"(disk-clear {_region_to_name(i)})")
    for a, b in _adjacent_pairs():
        init.append(f"(adjacent {_region_to_name(a)} {_region_to_name(b)})")
    goal = f"(object-at disk {_region_to_name(rg)})"
    path_disk_to_goal_cells = set(_bfs_path(rd, rg, set()))
    disk_zone = _expand_cells_by_1(path_disk_to_goal_cells)
    obstacles_on_path = [i for i in range(n_obstacles) if obstacles[i] in disk_zone]
    obstacle_list = ", ".join(f"obstacle{i}@{_region_to_name(obstacles[i])}" for i in range(n_obstacles))
    on_path_str = (
        f" Obstacles ON THE PATH (clear first): " + ", ".join(f"obstacle{i} at {_region_to_name(obstacles[i])}" for i in obstacles_on_path) + "."
    ) if obstacles_on_path else " No obstacles on direct disk→goal path; may still need clears."
    pickable_range = f"obstacle0..{n_pickable - 1}" if n_pickable > 0 else "none"
    pushonly_range = f"obstacle{n_pickable}..{n_obstacles - 1}" if n_pickable < n_obstacles else "none"
    scenario = (
        f"; robot={_region_to_name(re)}, disk={_region_to_name(rd)}, goal={_region_to_name(rg)}. Obstacles: {obstacle_list}. "
        f"{pickable_range} pickable (pick+place). " #; {pushonly_range} push-only (push_cube). "
        f"{on_path_str} "
        "Output: one line per subgoal, SKILL<TAB>STATE (one PDDL atom in parens)."
    )
    return f"""; PushO with obstacles — disk to goal.
{_region_layout_comment()}
{scenario}
(define (problem pusho-1)
  (:domain pusho)
  (:objects {objects})
  (:init
    {" ".join(init)}
  )
  (:goal {goal})
)"""


def _load_config():
    for path in [
        os.path.join(os.path.dirname(__file__), "config.json"),
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
    ]:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


def _parse_problem_regions(problem_str: str):
    robot = re.search(r"\(robot-at robot1 (r_\d+_\d+)\)", problem_str)
    disk = re.search(r"\(object-at disk (r_\d+_\d+)\)", problem_str)
    goal = re.search(r"\(goal-at (r_\d+_\d+)\)", problem_str)
    blocked = set(_name_to_idx(m.group(1)) for m in re.finditer(r"\(obstacle-at obstacle\d+ (r_\d+_\d+)\)", problem_str))
    return (
        _name_to_idx(disk.group(1)) if disk else 0,
        _name_to_idx(goal.group(1)) if goal else 0,
        _name_to_idx(robot.group(1)) if robot else 0,
        blocked,
    )


def _adjacency():
    adj = [[] for _ in range(GRID * GRID)]
    for a, b in _adjacent_pairs():
        adj[a].append(b)
    return adj


def _expand_cells_by_1(cells: set) -> set:
    """Return cells plus all their cardinal neighbors (disk ~3x3 needs 1-cell clearance)."""
    adj = _adjacency()
    expanded = set(cells)
    for cell in cells:
        for n in adj[cell]:
            expanded.add(n)
    return expanded


def _bfs_path(start: int, goal: int, blocked: set) -> list:
    from collections import deque

    adj = _adjacency()
    seen = {start}
    q = deque([(start, [])])
    while q:
        r, path = q.popleft()
        if r == goal:
            return path + [r]
        for n in adj[r]:
            if n not in seen and n not in blocked:
                seen.add(n)
                q.append((n, path + [r]))
    return []


def _parse_obstacle_info(problem_str: str) -> tuple[dict, set, set]:
    """Parse obstacle positions, pickable set, and push-only set from a problem string."""
    positions = {
        m.group(1): _name_to_idx(m.group(2))
        for m in re.finditer(r"\(obstacle-at (obstacle\d+) (r_\d+_\d+)\)", problem_str)
    }
    pickable  = {m.group(1) for m in re.finditer(r"\(pickable (obstacle\d+)\)", problem_str)}
    push_only = {m.group(1) for m in re.finditer(r"\(push-only (obstacle\d+)\)", problem_str)}
    return positions, pickable, push_only


def _clear_path_subgoals(disk_r: int, goal_r: int, blocked: set, problem_str: str) -> list[dict]:
    """
    When no disk→goal path exists around blocks, generate pick/place or push_cube
    subgoals to clear the blocks sitting on the direct (unblocked) disk→goal route.
    After these subgoals the executor re-plans, at which point a free path should exist.
    """
    direct_path = _bfs_path(disk_r, goal_r, set())
    if not direct_path:
        return []

    path_cells = set(direct_path)
    disk_zone = _expand_cells_by_1(path_cells)  # disk ~3x3: clear 1 cell around each path node
    positions, pickable, push_only = _parse_obstacle_info(problem_str)
    adj = _adjacency()

    occupied = set(positions.values())  # tracks all taken drop spots to avoid duplicates
    subgoals = []
    for obstacle_name, obstacle_region in positions.items():
        if obstacle_region not in disk_zone:
            continue

        # BFS outward from the obstacle to find the nearest cell that is
        # outside the disk zone, unblocked, not already occupied, and at least 2 steps away.
        drop = None
        seen_drop = {obstacle_region}
        frontier = [(obstacle_region, 0)]
        while frontier:
            cell, depth = frontier.pop(0)
            for n in adj[cell]:
                if n in seen_drop:
                    continue
                seen_drop.add(n)
                if n not in blocked and n not in disk_zone and n not in occupied and depth + 1 >= 2:
                    drop = n
                    break
                frontier.append((n, depth + 1))
            if drop is not None:
                break
        # Fallback: any unblocked, unoccupied adjacent cell if BFS found nothing
        if drop is None:
            drop = next((n for n in adj[obstacle_region] if n not in blocked and n not in occupied), None)
        if drop is not None:
            occupied.add(drop)

        subgoals.append({
            "skill": "reach",
            "state": f"(robot-at robot1 {_region_to_name(obstacle_region)})",
        })

        if obstacle_name in pickable:
            subgoals.append({"skill": "pick",  "state": f"(holding robot1 {obstacle_name})"})
            if drop is not None:
                subgoals.append({"skill": "place", "state": f"(obstacle-at {obstacle_name} {_region_to_name(drop)})"})
        elif obstacle_name in push_only and drop is not None:
            print("    [SKIP] push_cube skill disabled but cube is labeled as push_only, dangerously skipping")
        # elif obstacle_name in push_only and drop is not None:
        #     subgoals.append({"skill": "push_cube", "state": f"(obstacle-at {obstacle_name} {_region_to_name(drop)})"})

    return subgoals


def compute_subgoals(problem_str: str) -> list[dict]:
    disk_r, goal_r, robot_r, blocked = _parse_problem_regions(problem_str)
    disk_blocked = _expand_cells_by_1(blocked)  # disk ~3x3: block cells adjacent to any obstacle
    path_robot_to_disk = _bfs_path(robot_r, disk_r, blocked)
    path_disk_to_goal  = _bfs_path(disk_r, goal_r, disk_blocked)
    subgoals = []
    if path_robot_to_disk and path_robot_to_disk[-1] == disk_r:
        subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {_region_to_name(disk_r)})"})
    if path_disk_to_goal:
        for i in range(1, len(path_disk_to_goal)):
            subgoals.append({"skill": "push_disk", "state": f"(object-at disk {_region_to_name(path_disk_to_goal[i])})"})
    else:
        # All routes blocked — emit clearing subgoals for blocks near the direct path,
        # then append push_disk steps using a BFS path with those blocks removed.
        subgoals.extend(_clear_path_subgoals(disk_r, goal_r, blocked, problem_str))
        direct_path = _bfs_path(disk_r, goal_r, set())
        direct_zone = _expand_cells_by_1(set(direct_path))
        cleared_cells = {c for c in blocked if c in direct_zone}
        new_path = _bfs_path(disk_r, goal_r, _expand_cells_by_1(blocked - cleared_cells))
        if new_path and len(new_path) >= 2:
            subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {_region_to_name(disk_r)})"})
            for i in range(1, len(new_path)):
                subgoals.append({"skill": "push_disk", "state": f"(object-at disk {_region_to_name(new_path[i])})"})
    if not subgoals and path_disk_to_goal and path_disk_to_goal[0] != goal_r:
        subgoals.append({"skill": "push_disk", "state": f"(object-at disk {_region_to_name(goal_r)})"})
    return subgoals


def _push_disk_subgoals_only(problem_str: str) -> list[dict]:
    disk_r, goal_r, _, blocked = _parse_problem_regions(problem_str)
    path = _bfs_path(disk_r, goal_r, _expand_cells_by_1(blocked))
    if len(path) < 2:
        return []
    return [{"skill": "push_disk", "state": f"(object-at disk {_region_to_name(path[i])})"} for i in range(1, len(path))]


def run_pddl_planner(domain_path: str, problem_str: str, timeout: int = 10) -> str | None:
    # Create the temporary problem file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as f:
        f.write(problem_str)
        problem_path = f.name

    soln_path = problem_path + ".soln"
    try:
        subprocess.run(
            ["pyperplan", "-s", "gbf", "-H", "hff", domain_path, problem_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        if os.path.isfile(soln_path):
            with open(soln_path, "r") as f:
                plan = f.read().strip()
            return plan if plan else None
        return None
    except Exception as e:
        print(f"[pyperplan] Failed to generate plan: {e}")
        return None
    finally:
        for path in [problem_path, soln_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

def plan_to_subgoals(plan_str: str, _problem_str: str) -> list[dict]:
    subgoals = []
    for line in plan_str.splitlines():
        line = line.strip()
        if not line or not line.startswith("("):
            continue
        m = re.match(r"\(reach robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "reach", "state": f"(robot-at robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(push_disk robot1 r_\d+_\d+ (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "push_disk", "state": f"(object-at disk {m.group(1)})"})
            continue
        m = re.match(r"\(pick robot1 (obstacle\d+) r_\d+_\d+\)", line)
        if m:
            subgoals.append({"skill": "pick", "state": f"(holding robot1 {m.group(1)})"})
            continue
        m = re.match(r"\(place robot1 (obstacle\d+) (r_\d+_\d+)\)", line)
        if m:
            subgoals.append({"skill": "place", "state": f"(obstacle-at {m.group(1)} {m.group(2)})"})
            continue
        # m = re.match(r"\(push_cube robot1 (obstacle\d+) r_\d+_\d+ (r_\d+_\d+)\)", line)
        # if m:
        #     subgoals.append({"skill": "push_cube", "state": f"(obstacle-at {m.group(1)} {m.group(2)})"})
    return subgoals


def _parse_subgoals_response(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if not line or line.startswith(";"):
            continue
        m = re.match(r"Subgoal\s*\d+\s*:\s*(.+)", line, re.I)
        if m:
            line = m.group(1).strip()
        if "\t" in line:
            skill, _, state = line.partition("\t")
        else:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            skill, state = parts[0].strip(), parts[1].strip()
        skill = skill.lower().strip()
        state = state.strip()
        if state.startswith("(") and ")" in state and skill in VALID_SKILLS:
            out.append({"skill": skill, "state": state})
    return out


def _goal_region_from_problem(problem_str: str) -> str | None:
    m = re.search(r"\(goal-at (r_\d+_\d+)\)", problem_str)
    return m.group(1) if m else None


def _build_subgoal_prompt(domain: str, problem_str: str) -> str:
    goal_region = _goal_region_from_problem(problem_str) or "r_0_0"
    return f"""You decompose this Push-O task into ordered logical subgoals (milestones). The movable object is an O-shaped disk (circular puck); PDDL uses the constant `disk` for it.
Each line is one milestone: the SKILL that achieves it, then TAB, then the target state as ONE PDDL atom in parentheses.
When picking and placing the obstacles, try to move them a bit farther away from other obstacles as they may interfere with each other.

Rules:
- Order matters: earlier lines must be achievable before later ones.
- The disk is ~3x the diameter of an obstacle cube (~1.5-tile radius). push_disk requires not just the destination cell to be (clear ?to), but ALL cells within 1 tile of the entire disk path to be free. For example, if the disk is going to r_4_4, all obstacles between r_3_3 and r_5_5 could block it.
- End with: push_disk	(object-at disk {goal_region})
- Before that push chain: if obstacles block the path, clear them (pick/place), then reach to the disk's cell, then many push_disk steps (one grid cell per push_disk toward {goal_region}).
- Skills: reach, push_disk, pick, place. States: (robot-at robot1 r_i_j), (object-at disk r_i_j), (holding robot1 obstacleN), (obstacle-at obstacleN r_i_j).

Domain:
{domain}

Problem:
{problem_str}

Output ONLY lines: SKILL<TAB>STATE. No other text."""


def _call_vertex_subgoals(prompt: str, model: str, temperature: float, config: dict) -> str:
    project = os.environ.get("VERTEX_PROJECT") or config.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION") or config.get("VERTEX_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Set VERTEX_PROJECT in env or config.json for Gemini/VertexAI backend")
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except ImportError:
        raise RuntimeError("Install google-cloud-aiplatform: pip install google-cloud-aiplatform")
    vertexai.init(project=project, location=location)
    gen_model = GenerativeModel(model)
    response = gen_model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": 2048},
    )

    if not response.candidates or not response.candidates[0].content.parts:
        raise RuntimeError("LLM returned empty response")
    return response.candidates[0].content.parts[0].text.strip()


def _call_openai_subgoals(prompt: str, model: str, temperature: float, config: dict) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("OPENAI_API_KEY")
    base_url = "https://ai-gateway.andrew.cmu.edu" #os.environ.get("OPENAI_BASE_URL") or config.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("LLM returned empty response")
    return text.strip()


def _call_llm_subgoals(domain: str, problem_str: str, model: str, temperature: float, config: dict) -> list[dict]:
    prompt = _build_subgoal_prompt(domain, problem_str)
    backend = (os.environ.get("LLM_BACKEND") or config.get("LLM_BACKEND") or "").lower()
    # Infer backend from model name if not explicitly set
    if not backend:
        if model and model.startswith("gemini"):
            backend = "gemini"
        else:
            backend = "gemini-2.5-flash"
    if backend == "gemini":
        default_model = "gemini-2.5-flash"
        resolved_model = model if model and model.startswith("gemini") else default_model
        text = _call_vertex_subgoals(prompt, resolved_model, temperature, config)
    else:
        text = _call_openai_subgoals(prompt, model, temperature, config)
    return _parse_subgoals_response(text)


def _align_llm_to_symbolic(llm_steps: list[dict], symbolic_steps: list[dict]) -> list[dict]:
    """Sound order from symbolic plan; keep LLM's skill when a step's state matches."""
    if not symbolic_steps:
        return llm_steps
    out = []
    j = 0
    for g in llm_steps:
        while j < len(symbolic_steps) and symbolic_steps[j]["state"] != g["state"]:
            s = symbolic_steps[j]
            out.append({"skill": s["skill"], "state": s["state"]})
            j += 1
        if j < len(symbolic_steps) and symbolic_steps[j]["state"] == g["state"]:
            out.append({"skill": g["skill"], "state": g["state"]})
            j += 1
    while j < len(symbolic_steps):
        s = symbolic_steps[j]
        out.append({"skill": s["skill"], "state": s["state"]})
        j += 1
    return out


def _ensure_disk_goal_tail(subgoals: list[dict], problem_str: str) -> list[dict]:
    out = list(subgoals)
    goal_region = _goal_region_from_problem(problem_str)
    if not goal_region:
        return out
    goal_state = f"(object-at disk {goal_region})"
    if out and out[-1].get("state") == goal_state:
        return out
    seen = {s["state"] for s in out}
    for e in _push_disk_subgoals_only(problem_str):
        if e["state"] not in seen:
            out.append(e)
            seen.add(e["state"])
        if e["state"] == goal_state:
            break
    return out


def get_subgoals(
    domain_path: str,
    problem_str: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    offline: bool = False,
    use_llm_first: bool = False,
) -> list[dict]:
    """
    Default: LLM logical subgoals → merge with full symbolic plan (sound order/skills where needed).
    Fallbacks: symbolic plan only → BFS.

    use_llm_first=True: LLM only (+ optional push_disk tail), no pyperplan repair.
    SUBGOAL_PDDL_FIRST=1 or LLM_PLAN_PLANNER_FIRST: pyperplan first (old behavior).
    """
    if offline or os.environ.get("LLM_PLAN_OFFLINE"):
        return compute_subgoals(problem_str)

    def _env_on(name: str) -> bool:
        v = (os.environ.get(name) or "").strip().lower()
        return v in ("1", "true", "yes", "on")

    if _env_on("SUBGOAL_PDDL_FIRST") or _env_on("LLM_PLAN_PLANNER_FIRST"):
        plan = run_pddl_planner(domain_path, problem_str)
        if plan:
            return plan_to_subgoals(plan, problem_str)
        return compute_subgoals(problem_str)

    if use_llm_first:
        with open(domain_path, "r") as f:
            domain_text = f.read()
        config = _load_config()
        try:
            llm_steps = _call_llm_subgoals(domain_text, problem_str, model, temperature, config)
        except Exception as e:
            print("Exception during LLM Call:")
            print(e)
            llm_steps = []
        if not llm_steps:
            return compute_subgoals(problem_str)
        return _ensure_disk_goal_tail(llm_steps, problem_str)

    with open(domain_path, "r") as f:
        domain_text = f.read()
    config = _load_config()
    llm_steps: list[dict] = []
    try:
        llm_steps = _call_llm_subgoals(domain_text, problem_str, model, temperature, config)
    except Exception as e:
        print("Exception during LLM Call:")
        print(e)
        llm_steps = []

    if not llm_steps:
        plan = run_pddl_planner(domain_path, problem_str)
        if plan:
            return plan_to_subgoals(plan, problem_str)
        return compute_subgoals(problem_str)

    print("LLM subgoals generated successfully")
        
    llm_steps = _ensure_disk_goal_tail(llm_steps, problem_str)

    sym = compute_subgoals(problem_str)
    if sym:
        return _align_llm_to_symbolic(llm_steps, sym)

    return llm_steps


if __name__ == "__main__":
    from env_subgoal_runner import main

    main()

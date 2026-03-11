# Teleop Usage

## Scripts

1. Collect raw demos using keyboard teleoperation:
```
python scripts/collect_demos.py --quiet --env_id="PushT-WithExtraObject-v1" --record_dir="demos" --demo_id=0
```

2. Process raw demos to filter out zero action steps, save visualization videos:
```
python scripts/truncate_demos.py --env_id="PushT-WithExtraObject-v1" --record_dir="demos" --demo_id=0
python scripts/vis_demo.py --env_id="PushT-WithExtraObject-v1" --record_dir="demos" --demo_id=0
```
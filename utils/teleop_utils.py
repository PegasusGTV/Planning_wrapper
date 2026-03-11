import numpy as np


def get_currently_pressed_keys(viewer):
    """Check which keys are currently pressed"""
    assert viewer is not None, "Set --render-mode='human' when setting --action-type=teleop"
    pressed_keys = set()

    # Position control
    if viewer.window.key_down("i"): pressed_keys.add("i")  # -x: forward from camera perspective
    if viewer.window.key_down("k"): pressed_keys.add("k")  # +x: backward from camera perspective
    if viewer.window.key_down("j"): pressed_keys.add("j")  # -y: left from camera perspective
    if viewer.window.key_down("l"): pressed_keys.add("l")  # +y: right from camera perspective
    if viewer.window.key_down("u"): pressed_keys.add("u")  # -z: down
    if viewer.window.key_down("o"): pressed_keys.add("o")  # +z: up

    if viewer.window.key_down("q"): pressed_keys.add("q")

    return pressed_keys

def get_teleop_action(pressed_keys):
    # Initialize actions
    action = np.zeros(6)  # For pd_ee_target_delta_pose

    EE_TRANS_ACTION = 0.1

    # End-effector position control
    if "i" in pressed_keys: action[0] = -EE_TRANS_ACTION   # -x
    if "k" in pressed_keys: action[0] = EE_TRANS_ACTION    # x
    if "j" in pressed_keys: action[1] = -EE_TRANS_ACTION   # -y
    if "l" in pressed_keys: action[1] = EE_TRANS_ACTION    # y
    if "u" in pressed_keys: action[2] = -EE_TRANS_ACTION   # -z
    if "o" in pressed_keys: action[2] = EE_TRANS_ACTION    # z
    
    return action

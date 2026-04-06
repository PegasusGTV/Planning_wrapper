"""
utils/rigid_fit.py
==================
Rigid-body fitting and rotation utilities for reconstructing object poses
from predicted particle clouds.

Public API
----------
kabsch(source, target)
    Fit rotation R and translation t (numpy, single frame).
    source, target : (N, 3) float32
    Returns R (3,3), t (3,)

kabsch_batch(source, targets)
    Vectorised Kabsch over a trajectory.
    source  : (N, 3)          – fixed template
    targets : (T, N, 3)       – one cloud per timestep
    Returns Rs (T,3,3), ts (T,3)

rotation_matrix_to_quaternion(R)
    3×3 rotation matrix → unit quaternion [w, x, y, z].

quaternion_to_axis_angle(q)
    [w, x, y, z] quaternion → axis-angle 3-vector (radians).

axis_angle_to_quaternion(v)
    3-vector axis-angle → [w, x, y, z].

pca_orientation(pts)
    Estimate a rotation matrix from the principal axes of a point set.
    Useful as a template-free fallback for the gripper stick.

estimate_tcp_pose(hand_particles, hand_template=None)
    (T, n_hand, 3) → positions (T,3), quaternions (T,4)
    Uses Kabsch when template is provided; centroid + PCA otherwise.

estimate_block_pose(block_particles, block_template=None)
    (T, n_block, 3) → positions (T,3), quaternions (T,4)
"""

from __future__ import annotations

import numpy as np

# TCP offset in the panda_hand local frame, from the URDF:
#   <joint name="panda_hand_tcp_joint">
#       <origin xyz="0 0 0.15"/>
#   </joint>
TCP_LOCAL = np.array([0.0, 0.0, 0.15], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch algorithm
# ─────────────────────────────────────────────────────────────────────────────

def kabsch(source: np.ndarray, target: np.ndarray):
    """
    Find the rotation R and translation t that minimise
        ||target - (source @ R.T + t)||²
    using the Kabsch–Umeyama SVD method.

    Parameters
    ----------
    source : (N, 3) array  – template / reference points
    target : (N, 3) array  – observed / predicted points

    Returns
    -------
    R : (3, 3) rotation matrix
    t : (3,)   translation  (t = target_centroid - R @ source_centroid)
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    c_s = source.mean(0)
    c_t = target.mean(0)

    S = source - c_s
    T = target - c_t

    H = S.T @ T                      # (3, 3)
    U, _, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = (Vt.T @ D @ U.T).astype(np.float32)
    t = (c_t - R @ c_s).astype(np.float32)
    return R, t


def kabsch_batch(
    source: np.ndarray,       # (N, 3)  fixed template
    targets: np.ndarray,      # (T, N, 3)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Kabsch over T frames.

    Returns
    -------
    Rs : (T, 3, 3)
    ts : (T, 3)
    """
    T = targets.shape[0]
    Rs = np.empty((T, 3, 3), dtype=np.float32)
    ts = np.empty((T, 3),    dtype=np.float32)
    for i in range(T):
        R, t = kabsch(source, targets[i])
        Rs[i] = R
        ts[i] = t
    return Rs, ts


# ─────────────────────────────────────────────────────────────────────────────
# Rotation conversions
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses the numerically stable Shepperd method.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Unit quaternion [w, x, y, z] → axis-angle 3-vector (||v|| = angle in rad).
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(w))
    axis = q[1:4]
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if w < 0:
        axis = -axis
    return (axis / sin_half * angle).astype(np.float32)


def axis_angle_to_quaternion(v: np.ndarray) -> np.ndarray:
    """
    Axis-angle 3-vector → unit quaternion [w, x, y, z].
    """
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = v / angle
    w = np.cos(angle / 2.0)
    xyz = np.sin(angle / 2.0) * axis
    return np.array([w, *xyz], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two [w,x,y,z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternion) of [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def delta_quaternion(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Shortest-path delta quaternion: q_delta s.t. q_delta ⊗ q_current = q_target.
    Returns axis-angle 3-vector for use as a rotation action.
    """
    q_delta = quat_multiply(q_target, quat_conjugate(q_current))
    # Ensure shortest path
    if q_delta[0] < 0:
        q_delta = -q_delta
    return quaternion_to_axis_angle(q_delta)


# ─────────────────────────────────────────────────────────────────────────────
# PCA-based orientation (template-free fallback)
# ─────────────────────────────────────────────────────────────────────────────

def pca_orientation(pts: np.ndarray) -> np.ndarray:
    """
    Estimate a 3×3 rotation matrix from the principal axes of *pts*.

    The returned matrix maps world-frame axes to the estimated object axes:
      col-0 = smallest-variance axis  (thin axis)
      col-1 = middle-variance axis
      col-2 = largest-variance axis   (longest / stick axis)

    Parameters
    ----------
    pts : (N, 3)

    Returns
    -------
    R : (3, 3)  orthonormal rotation matrix
    """
    centered = pts - pts.mean(0)
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    R = eigenvectors.copy()

    # Make column orientation consistent: enforce det = +1
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    # Flip columns so the "stick" axis points roughly upward (+Z)
    for col in range(3):
        if R[2, col] < 0:
            R[:, col] *= -1

    return R.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# High-level pose estimators
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tcp_pose(
    hand_particles: np.ndarray,                   # (T, n_hand, 3)
    hand_template: np.ndarray | None = None,      # (n_hand, 3) in local frame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate gripper TCP pose from a trajectory of hand-particle clouds.

    The TCP is at TCP_LOCAL = [0, 0, 0.15] in the panda_hand local frame
    (from the URDF panda_hand_tcp_joint offset).  Given Kabsch fit (R, t),
    the world-frame TCP position is:

        tcp_world = R @ TCP_LOCAL + t

    This differs from using the cloud centroid whenever R is non-identity,
    with error proportional to ||TCP_LOCAL - centroid_local|| * sin(angle).

    Strategy:
      - With template : Kabsch fitting → exact rigid transform each frame.
      - Without template : centroid position + PCA orientation.
        (No TCP correction possible without the template — centroid used.)

    Returns
    -------
    positions  : (T, 3)  world-frame TCP position
    quaternions: (T, 4)  [w, x, y, z]
    """
    T = hand_particles.shape[0]
    positions   = np.empty((T, 3), dtype=np.float32)
    quaternions = np.empty((T, 4), dtype=np.float32)

    if hand_template is not None:
        Rs, ts = kabsch_batch(hand_template, hand_particles)
        for i in range(T):
            # TCP position in world: apply the same rigid transform to the
            # TCP anchor point in local frame (NOT the cloud centroid).
            positions[i]   = Rs[i] @ TCP_LOCAL + ts[i]
            quaternions[i] = rotation_matrix_to_quaternion(Rs[i])
    else:
        # No template — fall back to centroid + PCA, no TCP offset correction.
        for i in range(T):
            pts = hand_particles[i]
            positions[i]   = pts.mean(0)
            R              = pca_orientation(pts)
            quaternions[i] = rotation_matrix_to_quaternion(R)

    return positions, quaternions


def estimate_block_pose(
    block_particles: np.ndarray,                   # (T, n_block, 3)
    block_template: np.ndarray | None = None,      # (n_block, 3) in local frame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate block pose from a trajectory of block-particle clouds.

    The block has no special tool-center-point offset, so the centroid of
    the template (in local frame) is the natural pose anchor.

    When *block_template* is provided the result includes the rotation;
    otherwise only the centroid position is returned (identity rotation).

    Returns
    -------
    positions  : (T, 3)
    quaternions: (T, 4)  [w, x, y, z]
    """
    T = block_particles.shape[0]
    positions   = np.empty((T, 3), dtype=np.float32)
    quaternions = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (T, 1))

    if block_template is not None:
        Rs, ts = kabsch_batch(block_template, block_particles)
        c_local = block_template.mean(0)
        for i in range(T):
            positions[i]   = Rs[i] @ c_local + ts[i]
            quaternions[i] = rotation_matrix_to_quaternion(Rs[i])
    else:
        positions = block_particles.mean(axis=1)   # (T, 3)

    return positions, quaternions
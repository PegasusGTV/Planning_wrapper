; =============================================================================
; PDDL domain: PushO with obstacles (tabletop manipulation)
; =============================================================================
; TABLE: Grid of regions r_m_n (m=row, n=column). Goal = one of these regions.
; DISK SIZE: The disk is ~3x the diameter of an obstacle cube (~1.5-tile radius).
;   push_disk requires not just the destination cell to be (clear ?to), but ALL cells
;   within 1 tile of the entire disk path to be free. Clear every obstacle adjacent to
;   the planned push_disk corridor BEFORE executing push_disk steps.
; OBSTACLES: 10 obstacle cubes (obstacle0..obstacle9). Each obstacle has a SIZE relative to
;   the gripper/end-effector:
;   - (pickable ?b): obstacle is SMALLER than gripper → use PICK and PLACE to move
;     it out of the way.
;   - (push-only ?b): obstacle is BIGGER than gripper → cannot pick; use PUSH_CUBE
;     to push it one cell out of the way.
;   When picking and placing the obstacles, try to move them a bit farther away from other obstacles as they may interfere with each other.
; The planner must figure out from (pickable obstaclei) and (push-only obstaclej) which
; action to use for each obstacle. Main objective: get the disk to the goal region.
; =============================================================================

(define (domain pusho)
  (:requirements :strips :typing)
  (:types robot disk obstacle region)

  (:predicates
    (robot-at ?r - robot ?loc - region)
    (object-at ?o - disk ?loc - region)
    (obstacle-at ?b - obstacle ?loc - region)
    (holding ?r - robot ?b - obstacle)
    (hand-empty ?r - robot)
    (goal-at ?loc - region)
    (clear ?loc - region)
    (disk-clear ?loc - region)
    (adjacent ?loc1 ?loc2 - region)
    (pickable ?b - obstacle)
    (push-only ?b - obstacle))

  (:action reach
    :parameters (?rob - robot ?from ?to - region)
    :precondition (and (robot-at ?rob ?from) (adjacent ?from ?to))
    :effect (and (robot-at ?rob ?to) (not (robot-at ?rob ?from))))

  (:action push_disk
    :parameters (?rob - robot ?from ?to - region)
    :precondition (and (hand-empty ?rob) (robot-at ?rob ?from) (object-at disk ?from) (adjacent ?from ?to) (disk-clear ?to))
    :effect (and (object-at disk ?to) (not (object-at disk ?from)) (robot-at ?rob ?to) (not (robot-at ?rob ?from))))

  (:action pick
    :parameters (?rob - robot ?b - obstacle ?loc - region)
    :precondition (and (hand-empty ?rob) (pickable ?b) (robot-at ?rob ?loc) (obstacle-at ?b ?loc))
    :effect (and (holding ?rob ?b) (not (hand-empty ?rob)) (not (obstacle-at ?b ?loc)) (clear ?loc) (disk-clear ?loc)))

  (:action place
    :parameters (?rob - robot ?b - obstacle ?loc - region)
    :precondition (and (holding ?rob ?b) (robot-at ?rob ?loc) (clear ?loc))
    :effect (and (obstacle-at ?b ?loc) (not (holding ?rob ?b)) (hand-empty ?rob) (not (clear ?loc)) (not (disk-clear ?loc)))))
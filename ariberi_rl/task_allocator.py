"""Task allocation: assign objects to robots using distance + congestion."""

import config as cfg
from pathfinding import manhattan_distance


def compute_congestion(robot_pos, all_robot_positions, robot_id):
    """Compute congestion score for a robot based on nearby robots."""
    congestion = 0
    for i, other_pos in enumerate(all_robot_positions):
        if i == robot_id:
            continue
        dist = manhattan_distance(robot_pos, other_pos)
        if dist <= cfg.CONGESTION_RADIUS:
            congestion += 1.0 / max(dist, 1)
    return congestion


def allocate_tasks_greedy(robot_positions, available_objects, all_robot_positions=None,
                          use_congestion=False):
    """Assign objects to robots using greedy closest-first allocation.
    
    Args:
        robot_positions: dict {robot_id: (row, col)}
        available_objects: list of (row, col) for unassigned objects
        all_robot_positions: list of all robot positions (for congestion)
        use_congestion: whether to include congestion in cost
    
    Returns:
        dict {robot_id: (row, col) target} — robots without assignment are excluded
    """
    if not available_objects:
        return {}

    assignments = {}
    remaining_objects = list(available_objects)

    # Build cost matrix
    costs = []
    for rid, rpos in robot_positions.items():
        for obj in remaining_objects:
            dist = manhattan_distance(rpos, obj)
            cong = 0.0
            if use_congestion and all_robot_positions is not None:
                cong = compute_congestion(rpos, all_robot_positions, rid)
            cost = dist + cfg.CONGESTION_WEIGHT * cong
            costs.append((cost, rid, obj))

    costs.sort()

    assigned_robots = set()
    assigned_objects = set()

    for cost, rid, obj in costs:
        if rid in assigned_robots or obj in assigned_objects:
            continue
        assignments[rid] = obj
        assigned_robots.add(rid)
        assigned_objects.add(obj)
        if len(assignments) >= len(robot_positions):
            break

    return assignments


def reallocate_on_failure(assignments, failed_robot_id, robot_positions,
                          available_objects, all_robot_positions=None,
                          use_congestion=False):
    """Reassign failed robot's task to another available robot."""
    failed_target = assignments.pop(failed_robot_id, None)
    if failed_target is None:
        return assignments

    # Add the failed target back to available
    remaining = list(available_objects) + [failed_target]

    # Get robots that don't have assignments
    free_robots = {rid: pos for rid, pos in robot_positions.items()
                   if rid != failed_robot_id and rid not in assignments}

    if not free_robots:
        return assignments

    new_assignments = allocate_tasks_greedy(
        free_robots, [failed_target], all_robot_positions, use_congestion
    )
    assignments.update(new_assignments)
    return assignments

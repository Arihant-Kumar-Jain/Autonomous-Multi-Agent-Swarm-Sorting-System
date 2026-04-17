import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    algo_arg = DeclareLaunchArgument('algo', default_value='bfs', description='Algorithm to evaluate: bfs, q, or dqn')
    algo = LaunchConfiguration('algo')

    targets = [
        {'id': 0, 'tx': '2.0', 'ty': '2.0'},
        {'id': 1, 'tx': '-2.0', 'ty': '2.0'},
        {'id': 2, 'tx': '2.0', 'ty': '-2.0'}
    ]

    nodes = []
    
    # Depending on algo, pick the executable. In ROS2 launch, we can use a small Python trick or just start all with a remapped executable if we build it correctly. 
    # But since we can't easily dynamically change the executable string in pure LaunchConfiguration without OpaqueFunction, 
    # we'll use a wrapper node or just LaunchConfiguration to map strings.
    # Actually, LaunchConfiguration CAN be used as executable name if the executable name matches exactly.
    # Our executables will be: bfs_navigator, q_navigator, dqn_navigator.
    
    # We will name the executables `bfs`, `q`, `dqn` in setup.py, or just append `_navigator` here:
    # Actually, let's just make the user pass algo="bfs_navigator" or something, but we can also use OpaqueFunction.
    # Let's just assume executables are registered as bfs_navigator, q_navigator, dqn_navigator.
    from launch.substitutions import PythonExpression
    
    for t in targets:
        n = Node(
            package='eval_navigation',
            executable=PythonExpression(["'", algo, "_navigator'"]),
            name=f"nav_{t['id']}",
            parameters=[{
                'robot_id': t['id'],
                'target_x': float(t['tx']),
                'target_y': float(t['ty'])
            }],
            output='screen'
        )
        nodes.append(n)

    metrics = Node(
        package='eval_navigation',
        executable='metrics_node',
        parameters=[{'algo': algo}],
        output='screen'
    )
    nodes.append(metrics)

    return LaunchDescription([
        algo_arg,
        *nodes
    ])

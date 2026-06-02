"""
Launch file for Multi-Agent Warehouse Simulation.

Spawns:
  - Gazebo with warehouse world
  - 3 TurtleBot3 robots at spawn positions
  - Task coordinator node
  - Navigation node (BFS or RL)
  - Metrics logger

Usage:
  ros2 launch warehouse_multi_agent warehouse.launch.py mode:=bfs
  ros2 launch warehouse_multi_agent warehouse.launch.py mode:=rl
  ros2 launch warehouse_multi_agent warehouse.launch.py mode:=improved_rl
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    ExecuteProcess, GroupAction, TimerAction
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('warehouse_multi_agent')
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')

    # TurtleBot3 model — use 'burger' for lightweight simulation
    tb3_model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    # Launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode', default_value='bfs',
        description='Navigation mode: bfs, rl, or improved_rl'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='',
        description='Path to trained RL model (.pt file)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time'
    )

    world_file = os.path.join(pkg_dir, 'worlds', 'warehouse.world')

    # ─── Gazebo server + client ─────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'false',
        }.items()
    )

    # ─── Robot spawn positions (matching grid config) ───────────
    # Grid (row, col) → Gazebo (x=col, y=row) — 1:1 meter mapping
    robot_configs = [
        {'name': 'robot0', 'x': '1.0', 'y': '1.0', 'yaw': '0.0',
         'color': 'green', 'label': 'Atlas'},
        {'name': 'robot1', 'x': '7.0', 'y': '1.0', 'yaw': '0.0',
         'color': 'blue', 'label': 'Bolt'},
        {'name': 'robot2', 'x': '13.0', 'y': '1.0', 'yaw': '0.0',
         'color': 'red', 'label': 'Claw'},
    ]

    # ─── Spawn robots ──────────────────────────────────────────
    spawn_robots = []
    for i, rc in enumerate(robot_configs):
        # Each robot gets its own namespace
        ns = rc['name']

        # Get TurtleBot3 URDF
        tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
        urdf_file = os.path.join(
            get_package_share_directory('turtlebot3_description'),
            'urdf', f'turtlebot3_{tb3_model}.urdf'
        )

        # Robot state publisher (per robot)
        robot_state_pub = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=ns,
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_description': open(urdf_file).read() if os.path.exists(urdf_file) else '',
                'frame_prefix': f'{ns}/',
            }],
            output='screen',
        )

        # Spawn entity in Gazebo
        spawn = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', rc['name'],
                '-file', urdf_file if os.path.exists(urdf_file) else '',
                '-x', rc['x'],
                '-y', rc['y'],
                '-z', '0.01',
                '-Y', rc['yaw'],
                '-robot_namespace', ns,
            ],
            output='screen',
        )

        spawn_robots.append(robot_state_pub)
        spawn_robots.append(spawn)

    # ─── Task Coordinator Node ──────────────────────────────────
    coordinator = Node(
        package='warehouse_multi_agent',
        executable='task_coordinator.py',
        name='task_coordinator',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'num_robots': 3,
            'mode': LaunchConfiguration('mode'),
        }],
        output='screen',
    )

    # ─── Navigation Nodes (one per robot) ───────────────────────
    nav_nodes = []
    for i, rc in enumerate(robot_configs):
        # Choose navigator script based on mode
        # Both BFS and RL use the same rl_navigator.py which handles mode internally
        nav_node = Node(
            package='warehouse_multi_agent',
            executable='rl_navigator.py',
            name=f'navigator_{rc["name"]}',
            namespace=rc['name'],
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_id': i,
                'robot_name': rc['name'],
                'mode': LaunchConfiguration('mode'),
                'model_path': LaunchConfiguration('model_path'),
                'grid_scale': 1.0,  # 1 grid cell = 1 meter
            }],
            output='screen',
        )
        nav_nodes.append(nav_node)

    # ─── Metrics Logger ─────────────────────────────────────────
    metrics = Node(
        package='warehouse_multi_agent',
        executable='metrics_logger.py',
        name='metrics_logger',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'mode': LaunchConfiguration('mode'),
        }],
        output='screen',
    )

    # ─── Delayed start for coordinator (wait for robots) ────────
    delayed_coordinator = TimerAction(
        period=10.0,  # wait 10s for robots to spawn
        actions=[coordinator]
    )
    delayed_nav = TimerAction(
        period=12.0,
        actions=nav_nodes
    )
    delayed_metrics = TimerAction(
        period=13.0,
        actions=[metrics]
    )

    return LaunchDescription([
        mode_arg,
        model_path_arg,
        use_sim_time_arg,
        gazebo,
        *spawn_robots,
        delayed_coordinator,
        delayed_nav,
        delayed_metrics,
    ])

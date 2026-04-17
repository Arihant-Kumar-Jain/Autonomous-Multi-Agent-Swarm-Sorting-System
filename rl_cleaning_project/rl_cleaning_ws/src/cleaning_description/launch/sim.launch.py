import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('cleaning_description')
    world_file = os.path.join(pkg_share, 'worlds', 'simple_roads.world')
    urdf_file = os.path.join(pkg_share, 'urdf', 'simple_bot.urdf')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world_file}.items()
    )

    spawn_cmds = []
    nav_nodes = []
    
    # 3 robots starting on valid road coordinates
    starts = [(-4.9, 4.9), (0.1, 0.1), (4.9, -4.9)]
    
    for i in range(3):
        x, y = starts[i]
        name = f"robot_{i}"
        
        spawn = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', name,
                '-file', urdf_file,
                '-x', str(x),
                '-y', str(y),
                '-z', '0.1',
                '-robot_namespace', name
            ],
            output='screen'
        )
        spawn_cmds.append(spawn)
        
        nav = Node(
            package='cleaning_navigation',
            executable='rl_agent',
            name=f'rl_agent_{i}',
            parameters=[{
                'robot_id': i, 
                'model_path': '/home/aman/cs671_7/rl_cleaning_project/rl_training/models/q_table_R3.json'
            }],
            output='screen'
        )
        nav_nodes.append(nav)

    return LaunchDescription([
        gazebo,
        *spawn_cmds,
        *nav_nodes
    ])

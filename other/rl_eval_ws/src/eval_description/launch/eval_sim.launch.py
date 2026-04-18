import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('eval_description')
    world_path = os.path.join(pkg_share, 'worlds', 'proper_roads.world')
    urdf_path = os.path.join(pkg_share, 'urdf', 'agent.urdf')

    # Gazebo server and client
    gzserver = ExecuteProcess(
        cmd=['gzserver', world_path, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    spawn_entities = []
    # 3 robots starting at different corners
    starts = [
        {'id': 0, 'x': '-4.0', 'y': '0.0'},
        {'id': 1, 'x': '0.0', 'y': '-4.0'},
        {'id': 2, 'x': '4.0', 'y': '0.0'}
    ]

    for start in starts:
        node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', f"robot_{start['id']}",
                '-file', urdf_path,
                '-x', start['x'],
                '-y', start['y'],
                '-z', '0.1',
                '-robot_namespace', f"robot_{start['id']}"
            ],
            output='screen'
        )
        spawn_entities.append(node)

    return LaunchDescription([
        gzserver,
        gzclient,
        *spawn_entities
    ])

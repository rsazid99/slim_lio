"""
 * @file            launch/lio.launch.py
 * @description     
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-23 01:18:21
 * @lastModified    2026-04-23 03:30:30
 * Copyright ©Sazid Rahman Simanto All rights reserved
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_name = get_package_share_directory("slim_lio")
    config_file = os.path.join(pkg_name, 'config', 'mid360.yaml')
    rviz_config = os.path.join(pkg_name, 'rviz', 'slim.rviz')

    slim_lio = Node(
        package='slim_lio',
        executable='lio_node',
        name='lio_node',
        output='screen',
        parameters=[config_file]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        slim_lio,
        rviz_node,
    ])

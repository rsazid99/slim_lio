"""
 * @file            launch/lio.launch.py
 * @description     odometry only: lio_node (+ RViz in the odom frame)
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-04-23 03:41:43
 * @lastModified    2026-11-01 00:00:00
 * Copyright (c) 2026 Sazid Rahman Simanto
 * SPDX-License-Identifier: MIT
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('slim_lio')

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=os.path.join(share, 'config', 'colosseo.yaml'),
        description='lio_node parameter file')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='start RViz with rviz/lio.rviz (fixed frame odom)')

    lio_node = Node(
        package='slim_lio',
        executable='lio_node',
        name='lio_node',
        output='screen',
        parameters=[LaunchConfiguration('config')])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(share, 'rviz', 'lio.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([config_arg, rviz_arg, lio_node, rviz_node])

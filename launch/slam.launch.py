"""
 * @file            launch/slam.launch.py
 * @description     full SLAM: lio_node + slam_node (+ RViz in the map frame)
 * @author          rsazid99 <rsazid99@gmail.com>
 * @createTime      2026-11-01 00:00:00
 * @lastModified    2026-11-01 00:00:00
 * Copyright (c) 2026 Sazid Rahman Simanto
 * SPDX-License-Identifier: MIT
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
        description='start RViz with rviz/slim.rviz (fixed frame map)')

    # odometry front end, reusing lio.launch.py without its RViz. Launch configurations are
    # global, so the 'rviz': 'false' passed to the include would also disable our own RViz
    # below; the scoped GroupAction pops it again once the include has been processed.
    lio = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'lio.launch.py')),
            launch_arguments={'config': LaunchConfiguration('config'), 'rviz': 'false'}.items()),
    ], scoped=True)

    # pose-graph back end: loop closures, map -> odom TF, save_map service
    slam_node = Node(
        package='slim_lio',
        executable='slam_node',
        name='slam_node',
        output='screen')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(share, 'rviz', 'slim.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([config_arg, rviz_arg, lio, slam_node, rviz_node])

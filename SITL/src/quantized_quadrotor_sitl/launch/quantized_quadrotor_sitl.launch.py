from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_arg = DeclareLaunchArgument("config", default_value="configs/sitl_runtime.yaml")
    telemetry_node = Node(
        package="quantized_quadrotor_sitl",
        executable="telemetry_adapter_node",
        name="telemetry_adapter_node",
        parameters=[{"config_path": LaunchConfiguration("config")}],
        output="screen",
    )
    controller_node = Node(
        package="quantized_quadrotor_sitl",
        executable="controller_node",
        name="controller_node",
        parameters=[{"config_path": LaunchConfiguration("config")}],
        output="screen",
    )
    return LaunchDescription([config_arg, telemetry_node, controller_node])


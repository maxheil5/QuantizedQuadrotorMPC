from setuptools import find_packages, setup


package_name = "quantized_quadrotor_sitl"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/quantized_quadrotor_sitl.launch.py"]),
        (f"share/{package_name}/config", ["config/runtime.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Max",
    maintainer_email="max@example.com",
    description="Quantized Koopman/EDMD + MPC parity package and ROS 2 SITL nodes.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "offline_parity = quantized_quadrotor_sitl.experiments.offline_parity:main",
            "telemetry_adapter_node = quantized_quadrotor_sitl.ros.telemetry_adapter_node:main",
            "controller_node = quantized_quadrotor_sitl.ros.controller_node:main",
        ],
    },
)


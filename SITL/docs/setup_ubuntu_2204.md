# Ubuntu 22.04 Setup

This project targets Ubuntu 22.04 LTS with ROS 2 Humble, Gazebo Harmonic, and PX4 SITL. The required setup is non-default because ROS 2 Humble natively pairs with Gazebo Fortress, while this project intentionally uses Gazebo Harmonic.

## Official References Used

- Gazebo Harmonic ROS installation:
  - https://gazebosim.org/docs/harmonic/ros_installation/
- PX4 ROS 2 user guide:
  - https://docs.px4.io/main/en/ros2/user_guide.html
- PX4 Gazebo simulation:
  - https://docs.px4.io/main/en/sim_gazebo_gz/

## Important Conflict

The Gazebo docs explicitly note that ROS 2 Humble and Jazzy are not recommended with Harmonic unless you remove the default `ros_gz` packages and install the Harmonic-compatible variants instead. This workspace handles that conflict in `scripts/setup_linux.sh` instead of changing the requested stack.

## Automated Setup

Run:

```bash
cd SITL
./scripts/setup_linux.sh
```

The script performs these steps:

1. Verifies Ubuntu 22.04.
2. Installs ROS 2 Humble.
3. Installs Gazebo Harmonic from the OSRF package repository.
4. Removes conflicting `ros-humble-ros-gz*` packages if requested and installs `ros-humble-ros-gzharmonic`.
5. Creates a Python virtual environment and installs Python dependencies.
6. Clones PX4, `px4_msgs`, and `Micro-XRCE-DDS-Agent` inside `SITL/artifacts/` and `SITL/src/`.
7. Builds the Micro XRCE-DDS Agent.
8. Builds the ROS 2 workspace with `colcon`.

## Manual Setup Outline

If you do not want to use the script, the sequence is:

1. Install ROS 2 Humble on Ubuntu 22.04 from the official ROS repository.
2. Add the OSRF Gazebo repository and install `gz-harmonic`.
3. Remove default `ros-humble-ros-gz*` packages if they are installed.
4. Install `ros-humble-ros-gzharmonic`.
5. Clone PX4 and use the PX4 Ubuntu setup script with `--no-sim-tools` so PX4 does not overwrite the Gazebo stack you already selected.
6. Clone the matching `px4_msgs` branch into `SITL/src/px4_msgs`.
7. Build `Micro-XRCE-DDS-Agent`.
8. Build this workspace with `colcon build --symlink-install`.

## PX4 Message Versioning

The PX4 ROS 2 guide notes that message versioning matters. Use a `px4_msgs` branch that matches the PX4 checkout you are running, or add a translation layer if you intentionally mix versions.

## Runtime Order

For a typical SITL experiment:

1. Start the Micro XRCE-DDS Agent.
2. Start PX4 SITL in Gazebo Harmonic.
3. Source ROS 2 and the workspace install.
4. Launch the telemetry adapter and controller.


# Ubuntu External SSD Setup

This setup preserves the current machine by keeping the thesis runtime on a
bootable external SSD.

## Target

- Ubuntu `20.04.6 LTS`
- ROS `Noetic`
- Gazebo `11`
- catkin workspace at `~/thesis_v2_ws`
- repo checkout at `~/thesis_v2_ws/QuantizedQuadrotorMPC`

## External SSD install outline

1. Create a bootable Ubuntu `20.04.6` USB installer.
2. Connect the external SSD.
3. Install Ubuntu onto the external SSD only.
4. Do not modify the internal drive partition table.
5. Use the external SSD as the thesis runtime environment.

## First boot

After first boot on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl wget build-essential cmake pkg-config \
  python3-dev python3-pip python3-venv python3-rosdep python3-wstool \
  python3-catkin-tools liblapacke-dev protobuf-compiler libgoogle-glog-dev \
  libeigen3-dev
```

## ROS Noetic and Gazebo 11

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y ros-noetic-desktop-full ros-noetic-joy \
  ros-noetic-octomap-ros ros-noetic-mavlink ros-noetic-mavros \
  ros-noetic-control-toolbox
```

Initialize ROS dependency management:

```bash
sudo rosdep init || true
rosdep update
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Repo checkout

```bash
mkdir -p ~/thesis_v2_ws
cd ~/thesis_v2_ws
git clone <YOUR-REPO-REMOTE> QuantizedQuadrotorMPC
cd QuantizedQuadrotorMPC
```

## Bootstrap the workspace

Run:

```bash
bash "SITL V2/scripts/bootstrap_noetic_workspace.sh"
```

This script will:

- create `~/thesis_v2_ws/src`
- clone missing upstream repos
- symlink local V2 packages into the catkin workspace
- configure catkin for a Release build

## Freeze the environment once the first benchmark works

Run:

```bash
bash "SITL V2/scripts/freeze_environment.sh"
```

This writes a reproducibility snapshot under:

- `SITL V2/docs/environment_snapshots/`

## Notes

- ROS Noetic is an older ROS1 target, but it is the practical fit for
  RotorS + `mav_control_rw`.
- V2 deliberately avoids ROS 2 and PX4 dependencies.
- The current `SITL/` tree stays preserved and separate.


# koopman_mpc_ros

ROS1 wrapper package for the fresh V2 learned EDMD-MPC controller.

This package will subscribe to:

- `odometry`
- `command/pose`
- `command/trajectory`

and publish:

- `command/roll_pitch_yawrate_thrust`

The learned controller logic belongs in `koopman_python`. This package should
only handle ROS I/O, message translation, logging, and run orchestration.


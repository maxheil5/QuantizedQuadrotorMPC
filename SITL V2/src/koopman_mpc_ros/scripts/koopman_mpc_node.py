#!/usr/bin/env python3
"""ROS1 learned-controller node scaffold for SITL V2.

This node intentionally starts as a thin interface shell. The learned
controller math will live in the fresh `koopman_python` package.
"""

from __future__ import annotations


def main() -> None:
    try:
        import rospy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"rospy is required for koopman_mpc_ros: {exc}")

    rospy.init_node("koopman_mpc_node")
    rospy.loginfo("koopman_mpc_ros scaffold is installed. Controller logic is not implemented yet.")
    rospy.spin()


if __name__ == "__main__":
    main()


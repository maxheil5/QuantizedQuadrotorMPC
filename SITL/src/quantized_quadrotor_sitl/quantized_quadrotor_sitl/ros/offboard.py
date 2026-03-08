from __future__ import annotations


def vehicle_command_msg(command_type, timestamp: int, param1: float = 0.0, param2: float = 0.0):
    command = command_type()
    command.timestamp = timestamp
    command.param1 = float(param1)
    command.param2 = float(param2)
    command.target_system = 1
    command.target_component = 1
    command.source_system = 1
    command.source_component = 1
    command.from_external = True
    return command


def offboard_control_mode_msg(message_type, timestamp: int):
    msg = message_type()
    msg.timestamp = timestamp
    msg.position = False
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    msg.body_rate = False
    msg.thrust_and_torque = True
    msg.direct_actuator = False
    return msg


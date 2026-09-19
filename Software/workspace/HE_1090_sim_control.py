#!/usr/bin/env python3
import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Physical & Tuning Constants
WHEEL_RADIUS_M = 0.0255
CHASSIS_RADIUS_M = 0.06412
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])
_CTRL_LIMIT = 20.0

WAYPOINT_TOLERANCE = 0.12
CIRCLE_SEGMENTS = 72
CONTROL_PERIOD = 0.05

POS_KP = 0.8
POS_KI = 0.005
POS_KD = 0.04
POS_MAX_SPEED = 0.25
POS_INTEGRAL_CLAMP = 0.30

YAW_KP = 1.2
YAW_KI = 0.01
YAW_KD = 0.03
YAW_MAX_RATE = 0.8
YAW_DEAD_ZONE = math.radians(3.0)


def body_to_wheels(vx, vy, wz):
    wheel_speeds = (
        vx * np.cos(WHEEL_ANGLES_RAD)
        + vy * np.sin(WHEEL_ANGLES_RAD)
        + CHASSIS_RADIUS_M * wz
    ) / WHEEL_RADIUS_M
    return np.clip(wheel_speeds, -_CTRL_LIMIT, _CTRL_LIMIT).tolist()


def yaw_from_quat(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]
    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)
    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])
    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])
    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"Unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed

        self.pose = None
        self.start_pose = None
        self.wp_index = 0
        self.done = False

        self._prev_err_x = 0.0
        self._prev_err_y = 0.0
        self._prev_err_th = 0.0
        self._int_x = 0.0
        self._int_y = 0.0
        self._int_th = 0.0
        self._last_time = None

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/wheel_commands", 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.get_logger().info("Requesting shape from /get_shape...")
        shape_name, waypoints = self._request_shape()
        self.waypoints = waypoints
        self.get_logger().info(f"Shape assigned: {shape_name} ({len(waypoints)} waypoints)")

        self.timer = self.create_timer(CONTROL_PERIOD, self._control_step)

    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"get_shape failed: {response and response.message}")

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(q.w, q.x, q.y, q.z)
        self.pose = (x, y, yaw)

        if self.start_pose is None:
            self.start_pose = (x, y, yaw)
            self.get_logger().info(f"Start pose recorded: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f}°")

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def _control_step(self):
        if self.done or self.pose is None or self.start_pose is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = CONTROL_PERIOD
        if self._last_time is not None:
            dt = min(max(now - self._last_time, 1e-4), 0.2)
        self._last_time = now

        rx, ry, ryaw = self.pose

        if self.wp_index >= len(self.waypoints):
            if not self.done:
                self.done = True
                self._publish([0.0, 0.0, 0.0])
                if self.timer:
                    self.timer.cancel()
                self.get_logger().info("Shape completed! All waypoints reached – robot stopped.")
            return

        tx, ty = self.waypoints[self.wp_index]
        ex_w = tx - rx
        ey_w = ty - ry
        dist = math.sqrt(ex_w**2 + ey_w**2)

        if dist < WAYPOINT_TOLERANCE:
            self.wp_index += 1
            self.get_logger().info(f"Waypoint {self.wp_index}/{len(self.waypoints)} reached (dist={dist:.3f} m)")
            self._int_x = self._int_y = self._int_th = 0.0
            self._prev_err_x = self._prev_err_y = self._prev_err_th = 0.0
            self._publish([0.0, 0.0, 0.0])
            return

        self._int_x = float(np.clip(self._int_x + ex_w * dt, -POS_INTEGRAL_CLAMP, POS_INTEGRAL_CLAMP))
        self._int_y = float(np.clip(self._int_y + ey_w * dt, -POS_INTEGRAL_CLAMP, POS_INTEGRAL_CLAMP))

        d_ex = (ex_w - self._prev_err_x) / dt
        d_ey = (ey_w - self._prev_err_y) / dt
        self._prev_err_x = ex_w
        self._prev_err_y = ey_w

        vx_w = POS_KP * ex_w + POS_KI * self._int_x + POS_KD * d_ex
        vy_w = POS_KP * ey_w + POS_KI * self._int_y + POS_KD * d_ey

        max_spd = min(self.speed, POS_MAX_SPEED)
        spd = math.sqrt(vx_w**2 + vy_w**2)
        if spd > max_spd:
            scale = max_spd / spd
            vx_w *= scale
            vy_w *= scale

        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        vx_b = vx_w * cos_y + vy_w * sin_y
        vy_b = -vx_w * sin_y + vy_w * cos_y

        err_th = math.atan2(math.sin(self.start_pose[2] - ryaw), math.cos(self.start_pose[2] - ryaw))
        if abs(err_th) < YAW_DEAD_ZONE:
            err_th = 0.0

        self._int_th = float(np.clip(self._int_th + err_th * dt, -POS_INTEGRAL_CLAMP, POS_INTEGRAL_CLAMP))
        d_eth = (err_th - self._prev_err_th) / dt
        self._prev_err_th = err_th

        wz = float(np.clip(YAW_KP * err_th + YAW_KI * self._int_th + YAW_KD * d_eth, -YAW_MAX_RATE, YAW_MAX_RATE))

        self._publish(body_to_wheels(vx_b, vy_b, wz))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.25, help="max approach speed m/s (default 0.25)")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = None
    try:
        node = ShapeController(args.speed)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._publish([0.0, 0.0, 0.0])
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

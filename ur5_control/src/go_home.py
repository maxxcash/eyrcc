#!/usr/bin/env python3
"""
Move to Waypoint & Stop (Real Hardware)
---------------------------------------
1. Reads current position from /tcp_pose_raw
2. Moves to user-defined hardcoded waypoint.
3. Upon reaching, explicitly sends 0.0 velocity and exits.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys

class MoveToWaypoint(Node):
    def __init__(self):
        super().__init__('move_to_waypoint_node') 

        # ------------------ Configuration -------------------
        self.base_frame = 'base_link'
        
        # TARGET POSE (From your request)
        # Position: [x, y, z]
        self.target_pos = np.array([0.12039, -0.10902, 0.44477])
        
        # Orientation: [x, y, z, w]
        self.target_quat = np.array([0.50075, 0.49696, 0.50344, 0.49883])

        # Limits & Tolerances
        self.max_linear_speed = 0.05   # Increased slightly for single move (5 cm/s)
        self.max_angular_speed = 0.2   # ~11 deg/s
        self.pos_tolerance = 0.005     # 5mm accuracy
        self.orient_tolerance = 0.05   # ~2.8 degrees accuracy

        # ------------------ Communication -------------------
        # Publisher (Velocity Command)
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)

        # Subscriber (Current Pose Feedback)
        self.current_pos = None
        self.current_quat = None
        self.tcp_sub = self.create_subscription(
            Float64MultiArray, 
            '/tcp_pose_raw', 
            self.tcp_callback, 
            10
        )

        # Timer (Control Loop at 50Hz)
        self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info(" INITIALIZED: Waiting for TCP pose data...")

    def tcp_callback(self, msg):
        """Parse /tcp_pose_raw [x, y, z, rx, ry, rz]"""
        if len(msg.data) < 6:
            return
        
        # Update Position
        self.current_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])
        
        # Update Orientation (Euler XYZ -> Quaternion)
        try:
            r = R.from_euler('xyz', [msg.data[3], msg.data[4], msg.data[5]], degrees=False)
            self.current_quat = r.as_quat()
        except Exception as e:
            self.get_logger().warn(f"Rotation parsing error: {e}")

    def stop_arm(self):
        """Publishes explicit ZERO velocity to lock the arm."""
        stop_cmd = TwistStamped()
        stop_cmd.header.stamp = self.get_clock().now().to_msg()
        stop_cmd.header.frame_id = self.base_frame
        
        stop_cmd.twist.linear.x = 0.0
        stop_cmd.twist.linear.y = 0.0
        stop_cmd.twist.linear.z = 0.0
        stop_cmd.twist.angular.x = 0.0
        stop_cmd.twist.angular.y = 0.0
        stop_cmd.twist.angular.z = 0.0
        
        # Send multiple times to ensure hardware receives it
        for _ in range(5):
            self.twist_pub.publish(stop_cmd)
        
        self.get_logger().info(" ROBOT STOPPED: Zero velocity sent.")

    def control_loop(self):
        # 1. Safety: Do not move until we know where we are
        if self.current_pos is None or self.current_quat is None:
            return

        # 2. Calculate Error (P-Controller)
        pos_err = self.target_pos - self.current_pos
        
        current_rot = R.from_quat(self.current_quat)
        target_rot = R.from_quat(self.target_quat)
        
        # Quaternion difference
        rot_err = target_rot * current_rot.inv()
        rotvec = rot_err.as_rotvec()

        # 3. Check for Arrival
        dist_error = np.linalg.norm(pos_err)
        angle_error = np.linalg.norm(rotvec)

        if dist_error < self.pos_tolerance and angle_error < self.orient_tolerance:
            self.get_logger().info(f" TARGET REACHED (Err: {dist_error:.3f}m). Stopping.")
            self.stop_arm()
            # Clean exit
            raise SystemExit

        # 4. Compute Control Signals
        kp_pos = 0.8  # Gain for position
        kp_ang = 4.0  # Gain for orientation

        linear_cmd = kp_pos * pos_err
        angular_cmd = kp_ang * rotvec

        # 5. Safety: Clamp Velocities
        if np.linalg.norm(linear_cmd) > self.max_linear_speed:
            linear_cmd *= (self.max_linear_speed / np.linalg.norm(linear_cmd))
        
        if np.linalg.norm(angular_cmd) > self.max_angular_speed:
            angular_cmd *= (self.max_angular_speed / np.linalg.norm(angular_cmd))

        # 6. Publish Command
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = float(linear_cmd[0]), float(linear_cmd[1]), float(linear_cmd[2])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = float(angular_cmd[0]), float(angular_cmd[1]), float(angular_cmd[2])
        
        self.twist_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MoveToWaypoint()
    
    try:
        # Use simple spin; the node raises SystemExit when done
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("Main").info("Sequence Finished.")
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_arm()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

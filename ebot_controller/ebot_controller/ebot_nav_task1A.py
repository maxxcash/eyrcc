#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros # We only need the broadcaster part now
import math
import time

# --- Helper Functions ---
def quaternion_to_yaw(qx, qy, qz, qw):
    """Converts a quaternion into yaw (z-axis rotation)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """Normalize angle to be within the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class EbotNavigator(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- Publishers & Subscribers ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)

        # ### --- MODIFIED: We ONLY need the TF Broadcaster now --- ###
        # The listener is removed to avoid dependency on the broken TF tree.
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- Robot State (from /odom) ---
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.laser_ranges = []
        self.lidar_angle_min = 0.0
        self.lidar_angle_increment = 0.0
        self.odom_received = False
        self.lidar_received = False

        # --- Waypoints ---
        self.waypoints = [
            (-1.53, -1.95, 1.57),
            (-1.53, 1.24, 1.57), 
            (0.13,  1.24,  0.00),
            (0.13,  0.55, -1.57), # ADDED: New invisible waypoint
            (0.38, -3.32, -1.57)
        ]
        self.pos_tolerance = 0.15
        self.yaw_tolerance = math.radians(5.0)

        # --- High-Performance Control Gains ---
        self.k_linear = 2.0
        self.k_angular = 3.0
        self.k_rotate = 3.5
        self.k_rotate_gentle = 1.2 # MODIFIED: Reduced from 1.8 for an even softer rotation
        self.max_lin_speed = 0.7
        self.max_rot_speed = 3.0

        # --- Obstacle Avoidance Parameters ---
        self.safety_stop_dist = 0.30
        self.slowdown_dist = 0.7
        self.side_safety_dist = 0.28
        self.nudge_gain = 2.0

        # --- State Machine ---
        self.current_idx = 0
        self.state = 'ROTATE_TO_GOAL'
        self.final_push_start_time = None # Added for final push logic
        self.final_push_duration = 1.0   # Added for final push logic
        self.get_logger().info('Ebot Navigator (Manual Goal Following + TF Viz): INITIALIZED.')
        
        # --- Control Loop Timer ---
        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.odom_received = True

    def lidar_callback(self, msg: LaserScan):
        self.laser_ranges = msg.ranges
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_increment = msg.angle_increment
        self.lidar_received = True

    def broadcast_current_waypoint_tf(self):
        """Publishes the current waypoint for visualization in RViz."""
        if self.current_idx >= len(self.waypoints):
            return
        goal_x, goal_y, _ = self.waypoints[self.current_idx]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'waypoint_goal'
        t.transform.translation.x = goal_x
        t.transform.translation.y = goal_y
        self.tf_broadcaster.sendTransform(t)

    # Obstacle avoidance logic remains identical.
    def get_obstacle_modifiers(self):
        if not self.lidar_received or not self.laser_ranges:
            return 1.0, 0.0
        num_readings = len(self.laser_ranges)
        def get_min_dist(start_angle_deg, end_angle_deg):
            start_index = int(math.radians(start_angle_deg) / self.lidar_angle_increment) + num_readings // 2
            end_index = int(math.radians(end_angle_deg) / self.lidar_angle_increment) + num_readings // 2
            start_index = max(0, min(num_readings - 1, start_index))
            end_index = max(0, min(num_readings - 1, end_index))
            if start_index > end_index: start_index, end_index = end_index, start_index
            arc = self.laser_ranges[start_index:end_index+1]
            valid_readings = [r for r in arc if r > 0.1 and not (math.isinf(r) or math.isnan(r))]
            return min(valid_readings) if valid_readings else 100.0
        front_dist = get_min_dist(-15, 15)
        left_dist = get_min_dist(35, 65)
        right_dist = get_min_dist(-65, -35)
        linear_scale = 1.0
        if front_dist < self.safety_stop_dist:
            linear_scale = 0.0
        elif front_dist < self.slowdown_dist:
            linear_scale = (front_dist - self.safety_stop_dist) / (self.slowdown_dist - self.safety_stop_dist)
        angular_nudge = 0.0
        if left_dist < self.side_safety_dist:
            error = self.side_safety_dist - left_dist
            angular_nudge = -self.nudge_gain * error
        elif right_dist < self.side_safety_dist:
            error = self.side_safety_dist - right_dist
            angular_nudge = self.nudge_gain * error
        return linear_scale, angular_nudge

    def control_loop(self):
        if not self.odom_received or not self.lidar_received:
            return

        # Always broadcast the goal for visualization, even though we don't listen to it.
        self.broadcast_current_waypoint_tf()

        # This outer check is to handle the final push state correctly
        if self.state == 'DONE' or (self.current_idx >= len(self.waypoints) and self.state != 'FINAL_PUSH'):
            if self.state != 'DONE':
                self.get_logger().info("🎉 All waypoints reached. Mission Accomplished.")
                self.stop_robot()
                self.state = 'DONE'
            return

        cmd = Twist()
        
        # This check prevents an index error after the last waypoint is done
        if self.current_idx < len(self.waypoints):
            goal_x, goal_y, goal_yaw = self.waypoints[self.current_idx]
            distance_to_goal = math.hypot(goal_x - self.x, goal_y - self.y)
            world_angle_to_goal = math.atan2(goal_y - self.y, goal_x - self.x)
            angle_to_goal = normalize_angle(world_angle_to_goal - self.yaw) # Error relative to robot
            final_yaw_error = normalize_angle(goal_yaw - self.yaw)
        else:
            # Dummy values for when we are in FINAL_PUSH state
            distance_to_goal, angle_to_goal, final_yaw_error = 0, 0, 0
        
        # State machine logic uses these manually calculated values.
        if self.state == 'ROTATE_TO_GOAL' or self.state == 'ROTATE_FINAL':
            target_angle_error = angle_to_goal if self.state == 'ROTATE_TO_GOAL' else final_yaw_error
            if abs(target_angle_error) > self.yaw_tolerance:
                # Use a gentle rotation at the 3rd waypoint (index 2) to avoid getting stuck
                if self.state == 'ROTATE_FINAL' and self.current_idx == 2:
                    cmd.angular.z = self.k_rotate_gentle * target_angle_error
                else:
                    cmd.angular.z = self.k_rotate * target_angle_error
            else:
                self.stop_robot()
                if self.state == 'ROTATE_TO_GOAL':
                    self.state = 'MOVE_TO_GOAL'
                    # We keep this log silent to avoid clutter
                    # self.get_logger().info(f"➡ Aligned. Moving to WP-{self.current_idx + 1}.")
                else:
                    time.sleep(0.5)
                    
                    # Delayed and conditional logging
                    if self.current_idx == 0: # Just finished waypoint 1
                         self.get_logger().info(f"✅ Waypoint 1 achieved.")
                    elif self.current_idx == 2: # Just finished waypoint 3
                        self.get_logger().info(f"✅ Waypoint 2 achieved.")
                    elif self.current_idx == 4: # Just finished waypoint 5 (the final one)
                        self.get_logger().info(f"✅ Waypoint 3 achieved.")

                    self.current_idx += 1
                    
                    if self.current_idx < len(self.waypoints):
                        self.state = 'ROTATE_TO_GOAL'
                    else:
                        # Last waypoint reached, start the final push
                        self.get_logger().info("🏁 Final position aligned. Performing final push.")
                        self.state = 'FINAL_PUSH'
                        self.final_push_start_time = self.get_clock().now()

        elif self.state == 'MOVE_TO_GOAL':
            if distance_to_goal > self.pos_tolerance:
                base_linear_speed = self.k_linear * distance_to_goal
                base_angular_speed = self.k_angular * angle_to_goal
                linear_scale, angular_nudge = self.get_obstacle_modifiers()
                cmd.linear.x = base_linear_speed * linear_scale
                cmd.angular.z = base_angular_speed + angular_nudge
            else:
                self.stop_robot()
                self.state = 'ROTATE_FINAL'
                # Silent log
                # self.get_logger().info(f"📍 Position reached. Aligning to final yaw.")
        
        elif self.state == 'FINAL_PUSH':
            elapsed_time = (self.get_clock().now() - self.final_push_start_time).nanoseconds / 1e9
            if elapsed_time < self.final_push_duration:
                cmd.linear.x = 2.8  # Constant forward speed
            else:
                self.stop_robot()
                self.state = 'DONE' # Mission is now fully complete


        # Clamp speeds and publish
        cmd.linear.x = max(0.0, min(self.max_lin_speed, cmd.linear.x))
        cmd.angular.z = max(-self.max_rot_speed, min(self.max_rot_speed, cmd.angular.z))
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = EbotNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, stopping node.')
    finally:
        if rclpy.ok():
            node.stop_robot()
            node.destroy_node()
            rclpy.try_shutdown()

if __name__ == '__main__':
    main()
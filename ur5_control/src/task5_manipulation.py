#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Twist  # Updated
from control_msgs.msg import JointJog            # Updated
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from std_srvs.srv import SetBool 

import numpy as np
import time

class HybridArmControl(Node):
    def __init__(self):
        super().__init__('hybrid_arm_control')

        # --- 1. PUBLISHERS (Updated types) ---
        self.pub_twist = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)       
        self.pub_joint = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # --- 2. CONFIGURATION ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Fixed Joint Angles (Radians)
        self.aruco_pick = np.array([-4.9, -0.9, -2.122, -0.057, 1.570, 0.0])
        self.aruco_drop = np.array([-2.941, -2.1, -1.361, -1.265, 1.570, 0.15])
        self.home_pos = np.array([-3.14, -0.59, -2.49, -0.057, 1.57, 0.0])
        
        # Bad Fruit Locations
        self.bad_fruit_pick = np.array([-1.50, -1.20, -1.94, -1.557, 1.57, 0.0])
        self.bad_fruit_intermidiate = np.array([-0.577, -1.20, -1.94, -1.557, 1.57, 0.0])
        self.bad_fruit_drop = np.array([0.05, -2.22, -1.002, -1.557, 1.57, 0.0]) 

        # State Variables
        self.current_joints = np.zeros(6)
        self.joints_received = False
        
        # Force Monitoring
        self.current_force = 0.0
        self.force_sub = self.create_subscription(Float32, '/net_wrench', self.force_cb, 10)

        # Wait Logic
        self.wait_start_time = None
        self.is_waiting = False
        
        # Subscribe to know where we are
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # Magnet Service
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')

        # Start the Mission Loop (50 Hz)
        self.create_timer(0.02, self.mission_loop)
        self.step = 0
        self.get_logger().info("Hybrid Control Started.")

    def joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = np.array(msg.position[:6])
            self.joints_received = True

    def force_cb(self, msg):
        self.current_force = msg.data 

    def mission_loop(self):
        if not self.joints_received:
            return

        # --- PAUSE LOGIC ---
        if self.is_waiting:
            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed < 2.0: # Wait 2 seconds
                self.stop_robot()
                return
            else:
                self.is_waiting = False
                self.step += 1
                self.get_logger().info(f"Pause complete. Starting Step {self.step}...")

        # ==========================================================
        # SEQUENCE 1: ARUCO
        # ==========================================================
        if self.step == 0:
            if self.move_joints_to(self.aruco_pick):
                self.get_logger().info("Reached Pre-Pick. Pausing...")
                self.start_wait()

        elif self.step == 1:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.05):
                self.get_logger().info("Hovering ArUco. Pausing...")
                self.start_wait()

        elif self.step == 2:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.01):
                self._set_magnet(True) 
                if self.current_force > 10.0:
                    self.get_logger().info(f"Grasp SUCCESS (Force: {self.current_force:.1f}). Lifting...")
                    self.start_wait()
                else:
                    self.get_logger().warn(f"Grasp FAILED/WAITING (Force: {self.current_force:.1f})... Retrying...")

        elif self.step == 3:
            if self.move_joints_to(self.aruco_pick):
                self.get_logger().info("Lifted. Pausing...")
                self.start_wait()

        elif self.step == 4:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("At Home. Pausing...")
                self.start_wait()

        elif self.step == 5:
            if self.move_joints_to(self.aruco_drop):
                self._set_magnet(False) 
                self.get_logger().info("Dropped ArUco. Pausing...")
                self.start_wait()

        elif self.step == 6:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("Returned Home. Pausing before Bad Fruit...")
                self.start_wait()

        # ==========================================================
        # SEQUENCE 2: BAD FRUIT 1
        # ==========================================================
        elif self.step == 7:
            if self.move_joints_to(self.bad_fruit_pick):
                self.get_logger().info("In Position (Bad Fruit). Pausing...")
                self.start_wait()

        elif self.step == 8:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.01):
                self._set_magnet(True) 
                self.get_logger().info("Hovering Bad Fruit 1. Pausing...")
                self.start_wait()

        elif self.step == 9:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.05):      
                if self.current_force > 10.0:
                    self.get_logger().info(f"Picked Bad Fruit 1 (Force: {self.current_force:.1f})! Pausing...")
                    self.start_wait()
                else:
                    self.get_logger().warn(f"Waiting for grasp... Force: {self.current_force:.1f}")

        elif self.step == 10:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Lifted Bad Fruit 1. Pausing...")
                self.start_wait()
        
        elif self.step == 11:
            if self.move_joints_to(self.bad_fruit_drop): 
                self._set_magnet(False) 
                self.get_logger().info("Dropped Bad Fruit 1. Pausing...")
                self.start_wait()

        elif self.step == 12:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Cleared Bin. Pausing...")
                self.start_wait()

        elif self.step == 13:
            if self.move_joints_to(self.bad_fruit_pick):
                self.get_logger().info("Back to Pick Position. Pausing...")
                self.start_wait()

        # ==========================================================
        # SEQUENCE 3: BAD FRUIT 2
        # ==========================================================
        elif self.step == 14:
            if self.servo_to_target("1425_bad_fruit_2", z_offset=0.01):
                self._set_magnet(True)
                self.get_logger().info("Hovering Bad Fruit 2. Pausing...")
                self.start_wait()

        elif self.step == 15:
            if self.servo_to_target("1425_bad_fruit_2", z_offset=0.1):      
                if self.current_force > 10.0:
                    self.get_logger().info(f"Picked Bad Fruit 2 (Force: {self.current_force:.1f})! Pausing...")
                    self.start_wait()
                else:
                    self.get_logger().warn(f"Waiting for grasp... Force: {self.current_force:.1f}")

        elif self.step == 16:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Lifted Bad Fruit 2. Pausing...")
                self.start_wait()
        
        elif self.step == 17:
            if self.move_joints_to(self.bad_fruit_drop): 
                self._set_magnet(False)
                self.get_logger().info("Dropped Bad Fruit 2. Pausing...")
                self.start_wait()

        elif self.step == 18:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Cleared Bin. Pausing...")
                self.start_wait()

        elif self.step == 19:
            if self.move_joints_to(self.bad_fruit_pick):
                self.get_logger().info("Back to Pick Position. Pausing...")
                self.start_wait()

        # ==========================================================
        # SEQUENCE 4: BAD FRUIT 3
        # ==========================================================
        elif self.step == 20:
            if self.servo_to_target("1425_bad_fruit_3", z_offset=0.01):
                self._set_magnet(True)
                self.get_logger().info("Hovering Bad Fruit 3. Pausing...")
                self.start_wait()

        elif self.step == 21:
            if self.servo_to_target("1425_bad_fruit_3", z_offset=0.1):      
                if self.current_force > 10.0:
                    self.get_logger().info(f"Picked Bad Fruit 3 (Force: {self.current_force:.1f})! Pausing...")
                    self.start_wait()
                else:
                    self.get_logger().warn(f"Waiting for grasp... Force: {self.current_force:.1f}")

        elif self.step == 22:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Lifted Bad Fruit 3. Pausing...")
                self.start_wait()
        
        elif self.step == 23:
            if self.move_joints_to(self.bad_fruit_drop): 
                self._set_magnet(False)
                self.get_logger().info("Dropped Bad Fruit 3. Pausing...")
                self.start_wait()

        elif self.step == 24:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.get_logger().info("Cleared Bin. Pausing...")
                self.start_wait()

        elif self.step == 25:
            if self.move_joints_to(self.bad_fruit_pick):
                self.get_logger().info("Back to Pick Position. Pausing...")
                self.start_wait()

        elif self.step == 26:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("All Missions Complete.")
                self.step = 27 

        elif self.step == 27:
            self.stop_robot()

    def start_wait(self):
        self.stop_robot()
        self.wait_start_time = self.get_clock().now()
        self.is_waiting = True

    # --- UPDATED JOINT MOVE (Send JointJog) ---
    def move_joints_to(self, target_array):
        error = target_array - self.current_joints
        
        if np.max(np.abs(error)) < 0.03:
            return True 
        
        Kp = 5.0
        cmd = error * Kp
        cmd = np.clip(cmd, -3.14, 3.14) 
        
        # Create JointJog message
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # Optional: You can add joint names here if your controller requires them
        # msg.joint_names = ['shoulder_pan_joint', ...]
        msg.velocities = cmd.tolist()
        msg.duration = 0.02 # Valid for 20ms
        
        self.pub_joint.publish(msg)
        return False

    # --- UPDATED CARTESIAN MOVE (Send TwistStamped) ---
    def servo_to_target(self, frame_name, y_offset=0.0, z_offset=0.0):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', frame_name, rclpy.time.Time())
            ee_trans = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time())
            
            target_pos = np.array([
                trans.transform.translation.x, 
                trans.transform.translation.y + y_offset, 
                trans.transform.translation.z + z_offset
            ])
            
            current_pos = np.array([
                ee_trans.transform.translation.x, 
                ee_trans.transform.translation.y, 
                ee_trans.transform.translation.z
            ])
            
            error = target_pos - current_pos
            distance = np.linalg.norm(error)

            if distance < 0.03: 
                return True
            
            # Create TwistStamped message
            ts = TwistStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = "base_link"
            ts.twist.linear.x = error[0] * 2.0
            ts.twist.linear.y = error[1] * 2.0
            ts.twist.linear.z = error[2] * 2.0
            
            self.pub_twist.publish(ts)
            return False

        except Exception:
            return False

    def stop_robot(self):
        # Stop Twist
        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = "base_link"
        self.pub_twist.publish(ts)
        
        # Stop Joints
        jj = JointJog()
        jj.header.stamp = self.get_clock().now().to_msg()
        jj.header.frame_id = "base_link"
        jj.velocities = [0.0] * 6
        self.pub_joint.publish(jj)

    # --- HELPER: MAGNET ---
    def _set_magnet(self, state):
        req = SetBool.Request()
        req.data = state
        self.magnet_cli.call_async(req)
        status = "ON" if state else "OFF"
        self.get_logger().info(f"Magnet set to {status}")

def main(args=None):
    rclpy.init(args=args)
    node = HybridArmControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

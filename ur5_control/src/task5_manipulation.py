#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# --- HARDWARE IMPORTS ---
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import numpy as np
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener

class HybridArmControl(Node):
    def __init__(self):
        super().__init__('hybrid_arm_control')

        # --- 1. PUBLISHERS ---
        self.pub_twist = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)       
        self.pub_joint = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # --- 2. SUBSCRIBERS ---
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(Float32, '/net_wrench', self.force_cb, 10)
        self.tcp_sub = self.create_subscription(Float64MultiArray, '/tcp_pose_raw', self.tcp_cb, 10)
        
        # --- 3. STATE VARIABLES ---
        self.current_joints = np.zeros(6)
        self.current_tcp_pos = np.zeros(3)
        self.current_force_z = 0.0
        self.joints_received = False
        
        # --- 4. WAIT/DELAY LOGIC ---
        self.step = 0
        self.is_waiting = False
        self.wait_start_time = None
        self.delay_duration = 1.0  # 1 Second delay
        self.next_step_after_wait = 0

        # --- 5. CONFIGURATION & TARGETS ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Waypoints
        self.aruco_pick = np.array([-4.9, -0.9, -2.122, -0.057, 1.570, 0.0])
        self.aruco_drop = np.array([-2.941, -2.120, -1.2, -1.390, 1.570, 0.15])
        self.home_pos = np.array([-3.14, -0.59, -2.49, -0.057, 1.57, 0.0])
        self.bad_fruit_pick = np.array([-1.50, -1.20, -1.94, -1.557, 1.57, 0.0])
        self.bad_fruit_intermidiate = np.array([-0.470, -1.95, -1.1, -1.469, 1.57, 0.0])
        self.bad_fruit_drop = np.array([0.06, -2.10, -1.2, -1.469, 1.57, 0.0]) 

        # --- 6. MAGNET SERVICE ---
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Magnet service...')

        # Main Loop Timer
        self.create_timer(0.02, self.mission_loop)
        self.get_logger().info("Hybrid Control Initialized with 1s Waypoint Delays.")

    # --- CALLBACKS ---
    def joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = np.array(msg.position[:6])
            self.joints_received = True

    def force_cb(self, msg):
        self.current_force_z = msg.data
        self.get_logger().info(f'LIVE FORCE Z: {self.current_force_z:.3f}', throttle_duration_sec=0.1)

    def tcp_cb(self, msg):
        self.current_tcp_pos = np.array(msg.data[:3])

    # --- HELPER: NON-BLOCKING WAIT ---
    def start_wait(self, next_step):
        """Stops the robot and starts a 1-second countdown."""
        self.stop_robot()
        self.is_waiting = True
        self.wait_start_time = self.get_clock().now()
        self.next_step_after_wait = next_step
        self.get_logger().info(f"Waypoint reached. Pausing for {self.delay_duration}s...")

    def mission_loop(self):
        if not self.joints_received:
            return

        # Check if we are currently in a waiting period
        if self.is_waiting:
            now = self.get_clock().now()
            elapsed = (now - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= self.delay_duration:
                self.is_waiting = False
                self.step = self.next_step_after_wait
            return # Skip movement logic while waiting

        # ==========================================================
        # MISSION SEQUENCE
        # ==========================================================
        
        # --- ARUCO SEQUENCE ---
        if self.step == 0:
            if self.move_joints_to(self.aruco_pick):
                self.start_wait(1)

        elif self.step == 1:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.01):
                self.control_magnet(True)
                self.start_wait(2)

        elif self.step == 2:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.0):
                self.get_logger().info(f"At Fertilizer. Force: {self.current_force_z:.2f}") 
                self.start_wait(3)

        elif self.step == 3:
            if self.servo_to_target("1425_fertilizer_1", z_offset=0.005):
                self.start_wait(4)

        elif self.step == 4:
            if self.move_joints_to(self.aruco_pick):
                self.start_wait(5)

        elif self.step == 5:
            if self.move_joints_to(self.home_pos):
                self.start_wait(6)

        elif self.step == 6:
            if self.move_joints_to(self.aruco_drop):
                self.start_wait(7)

        elif self.step == 7:
            if self.servo_to_target("1425_bot", z_offset=0.1):
                self.control_magnet(False)
                self.start_wait(8)

        elif self.step == 8:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("Starting Bad Fruit Mission...")
                self.start_wait(9)

        # --- BAD FRUIT 1 ---
        elif self.step == 9:
            if self.move_joints_to(self.bad_fruit_pick):
                self.start_wait(10)

        elif self.step == 10:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.02):
                self.control_magnet(True)
                self.start_wait(11)

        elif self.step == 11:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.0):
                self.start_wait(12)
        
        elif self.step == 12:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.02):
                self.start_wait(13)

        elif self.step == 13:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(14)
        
        elif self.step == 14:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.start_wait(15)

        # --- BAD FRUIT 2 ---
        elif self.step == 15:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(16)

        elif self.step == 16:
            if self.move_joints_to(self.bad_fruit_pick):
                self.start_wait(17)

        elif self.step == 17:
             if self.servo_to_target("1425_bad_fruit_2", z_offset=0.02):
                self.control_magnet(True)
                self.start_wait(18)
        
        elif self.step == 18:
            if self.servo_to_target("1425_bad_fruit_2", z_offset=0.0):  
                self.start_wait(19)

        elif self.step == 19:
            if self.servo_to_target("1425_bad_fruit_2", z_offset=0.02):    
                self.start_wait(20)

        elif self.step == 20:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(21)
        
        elif self.step == 21:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.start_wait(22)

        # --- BAD FRUIT 3 ---
        elif self.step == 22:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(23)

        elif self.step == 23:
            if self.move_joints_to(self.bad_fruit_pick):
                self.start_wait(24)

        elif self.step == 24:
             if self.servo_to_target("1425_bad_fruit_3", z_offset=0.02):
               self.control_magnet(True)
               self.start_wait(25)

        elif self.step == 25:
             if self.servo_to_target("1425_bad_fruit_3", z_offset=0.0):   
                self.start_wait(26)
        
        elif self.step == 26:
            if self.servo_to_target("1425_bad_fruit_3", z_offset=0.02):    
                self.start_wait(27)

        elif self.step == 27:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(28)
        
        elif self.step == 28:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.start_wait(29)
        
        elif self.step == 29:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.start_wait(30)

        elif self.step == 30:
            if self.move_joints_to(self.bad_fruit_pick):
                self.start_wait(31)

        elif self.step == 31:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("All Missions Complete.")
                self.step = 32  

        elif self.step == 32:
            self.stop_robot()

    # --- CONTROL HELPERS ---
    def move_joints_to(self, target_array):
        error = target_array - self.current_joints
        if np.max(np.abs(error)) < 0.03:
            return True 
        
        Kp = 2.0  
        cmd_vels = np.clip(error * Kp, -0.1, 0.1) 
        
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        msg.velocities = cmd_vels.tolist()
        self.pub_joint.publish(msg)
        return False

    def servo_to_target(self, frame_name, y_offset=0.0, z_offset=0.0):
        try:
            trans = self.tf_buffer.lookup_transform('base_link', frame_name, rclpy.time.Time())
            target_pos = np.array([
                trans.transform.translation.x, 
                trans.transform.translation.y - y_offset, 
                trans.transform.translation.z + z_offset
            ])
            
            error = target_pos - self.current_tcp_pos
            distance = np.linalg.norm(error)

            if distance < 0.03:
                return True
            
            vel = np.clip(error * 1.0, -0.1, 0.1)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = "base_link"
            twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z = vel
            self.pub_twist.publish(twist_msg)
            return False
        except Exception:
            return False

    def stop_robot(self):
        self.pub_twist.publish(TwistStamped())
        stop_joints = JointJog()
        stop_joints.joint_names = self.joint_names
        stop_joints.velocities = [0.0] * 6
        self.pub_joint.publish(stop_joints)

    def control_magnet(self, activate=True):
        req = SetBool.Request()
        req.data = activate
        self.magnet_cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = HybridArmControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

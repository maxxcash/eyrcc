#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# --- HARDWARE IMPORTS ---
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32  # <--- NEW: For Force Monitoring
from tf2_ros import Buffer, TransformListener
import numpy as np
from std_msgs.msg import Float64MultiArray

class HybridArmControl(Node):
    def __init__(self):
        super().__init__('hybrid_arm_control')

        # --- 1. PUBLISHERS ---
        self.pub_twist = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)       
        self.pub_joint = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # --- 2. SUBSCRIBERS ---
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # [NEW] Force Monitoring Subscriber
        # Based on your screenshot: Topic /net_wrench, Type Float32
        self.create_subscription(Float32, '/net_wrench', self.force_cb, 10)
        self.current_tcp_pos = np.zeros(3)
        self.tcp_sub = self.create_subscription(Float64MultiArray, '/tcp_pose_raw', self.tcp_cb, 10)
        # --- 3. CONFIGURATION ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Fixed Joint Angles
        self.aruco_pick = np.array([-4.9, -0.9, -2.122, -0.057, 1.570, 0.0])
        self.aruco_drop = np.array([-2.941, -2.120, -1.2, -1.390, 1.570, 0.15])
        self.aruco = np.array([-4.88, -1.8, -1.25, 0.01, 1.55, 0.0])
        self.home_pos = np.array([-3.14, -0.59, -2.49, -0.057, 1.57, 0.0])
        
        self.bad_fruit_pick = np.array([-1.50, -1.20, -1.94, -1.557, 1.57, 0.0])
        self.bad_fruit_intermidiate = np.array([-0.470, -1.95, -1.1, -1.469, 1.57, 0.0])
        self.bad_fruit_drop = np.array([0.15, -2.15, -1.2, -1.469, 1.57, 0.0]) 

        # State Variables
        self.current_joints = np.zeros(6)
        self.joints_received = False
        self.current_force_z = 0.0  # [NEW] Store latest force reading

        # --- 4. MAGNET SERVICE ---
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')

        # Start Loop
        self.create_timer(0.02, self.mission_loop)
        self.step = 0
        self.get_logger().info("Hardware Control Started (with Force Monitoring).")

    def joint_cb(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = np.array(msg.position[:6])
            self.joints_received = True

    # [NEW] Force Callback
    def force_cb(self, msg):
        # The message is a simple Float32 representing Z-force
        self.current_force_z = msg.data
        # Uncomment below to debug force values live in terminal:
        # self.get_logger().info(f'Force Z: {self.current_force_z:.2f}', throttle_duration_sec=1.0)
    
    def tcp_cb(self, msg):
        # msg.data contains [x, y, z, rx, ry, rz]
        self.current_tcp_pos = np.array(msg.data[:3])

    def mission_loop(self):
        if not self.joints_received:
            return

        # ==========================================================
        # SEQUENCE 1: ARUCO
        # ==========================================================
        if self.step == 0:
            if self.move_joints_to(self.aruco_pick):
                self.get_logger().info("Reached Pre-Pick. Looking for ArUco...")
                self.step = 1

        elif self.step == 1:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.1):
                self.get_logger().info("Hovering ArUco. Descending...")
                self.step = 2

        elif self.step == 2:
            if self.servo_to_target("1425_fertilizer_1", y_offset=0.0):
                self.control_magnet(True)
                # [NEW] Log force to verify contact
                self.get_logger().info(f"Magnet ON. Current Force: {self.current_force_z:.2f}") 
                self.step = 3

        # elif self.step == 1:
        #     if self.move_joints_to(self.aruco_pick):
        #        self.step = 2

        # elif self.step == 2:
        #     if self.move_joints_to(self.aruco):
        #         self.step = 3

        elif self.step == 3:
            if self.move_joints_to(self.aruco_pick):
                # [NEW] Verify payload weight
                self.get_logger().info(f"Lifted. Payload Check (Force): {self.current_force_z:.2f}")
                self.step = 4


        elif self.step == 4:
            if self.move_joints_to(self.home_pos):
                self.step = 5

        elif self.step == 5:
            if self.move_joints_to(self.aruco_drop):
                self.control_magnet(False)
                self.get_logger().info("Dropped ArUco.")
                self.step = 6

        elif self.step == 6:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("Home. Starting Bad Fruit...")
                self.step = 7

        # ==========================================================
        # SEQUENCE 2: BAD FRUIT 
        # ==========================================================
        elif self.step == 7:
            if self.move_joints_to(self.bad_fruit_pick):
                self.step = 8

        elif self.step == 8:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.1):
                self.step = 9

        elif self.step == 9:
            if self.servo_to_target("1425_bad_fruit_1", z_offset=0.0):
                self.control_magnet(True)
                self.get_logger().info(f"Magnet ON (Fruit 1). Force: {self.current_force_z:.2f}")
                self.step = 10

        elif self.step == 10:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 11
        
        elif self.step == 11:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.step = 12

        elif self.step == 12:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 13

        elif self.step == 13:
            if self.move_joints_to(self.bad_fruit_pick):
                self.step = 14

        # ... (Bad Fruit 2 & 3 - Logic remains same) ...

        elif self.step == 14:
             if self.servo_to_target("1425_bad_fruit_2", z_offset=0.1):
                self.step = 15
        
        elif self.step == 15:
            if self.servo_to_target("1425_bad_fruit_2", z_offset=0.0):  
                self.control_magnet(True)
                self.get_logger().info(f"Magnet ON (Fruit 2). Force: {self.current_force_z:.2f}")    
                self.step = 16

        elif self.step == 16:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 17
        
        elif self.step == 17:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.step = 18

        elif self.step == 18:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 19

        elif self.step == 19:
            if self.move_joints_to(self.bad_fruit_pick):
                self.step = 20

        # ... Bad Fruit 3 ...
        elif self.step == 20:
             if self.servo_to_target("1425_bad_fruit_3", z_offset=0.1):
               self.step = 21

        elif self.step == 21:
             if self.servo_to_target("1425_bad_fruit_3", z_offset=0.0):   
                self.control_magnet(True)
                self.get_logger().info(f"Magnet ON (Fruit 3). Force: {self.current_force_z:.2f}")   
                self.step = 22
        
        elif self.step == 22:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 23
        
        elif self.step == 23:
            if self.move_joints_to(self.bad_fruit_drop): 
                self.control_magnet(False)
                self.step = 24
        
        elif self.step == 24:
            if self.move_joints_to(self.bad_fruit_intermidiate): 
                self.step = 25

        elif self.step == 25:
            if self.move_joints_to(self.bad_fruit_pick):
                self.step = 26

        elif self.step == 26:
            if self.move_joints_to(self.home_pos):
                self.get_logger().info("All Missions Complete.")
                self.step = 27  

        elif self.step == 27:
            self.stop_robot()

    # --- HELPER: JOINT MOVE ---
    def move_joints_to(self, target_array):
        error = target_array - self.current_joints
        
        if np.max(np.abs(error)) < 0.03:
            self.stop_robot()
            return True 
        
        Kp = 2.0  
        cmd_vels = error * Kp
        cmd_vels = np.clip(cmd_vels, -0.1, 0.1) 
        
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = self.joint_names
        msg.velocities = cmd_vels.tolist()
        
        self.pub_joint.publish(msg)
        return False

    # --- HELPER: CARTESIAN MOVE ---
    # --- HELPER: CARTESIAN MOVE (With Speed Control) ---
    def servo_to_target(self, frame_name, y_offset=0.0, z_offset=0.0):
        try:
            # Get target from your perception TF
            trans = self.tf_buffer.lookup_transform('base_link', frame_name, rclpy.time.Time())
            
            target_pos = np.array([
                trans.transform.translation.x, 
                trans.transform.translation.y + y_offset, 
                trans.transform.translation.z + z_offset
            ])

            # Use TCP Pose as current position instead of lookup_transform for wrist_3
            current_pos = self.current_tcp_pos
            
            error = target_pos - current_pos
            distance = np.linalg.norm(error)

            if distance < 0.01: # You can now use a tighter tolerance (1cm)
                self.stop_robot()
                return True
            
            # 4. Calculate Raw Velocity (P-Controller)
            vel_x = error[0] * 1.0
            vel_y = error[1] * 1.0
            vel_z = error[2] * 1.0

            # 5. Speed Clipping (Safety Limit)
            current_speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            if current_speed > 0.1:
                scale_factor = 0.1 / current_speed
                vel_x *= scale_factor
                vel_y *= scale_factor
                vel_z *= scale_factor

            # 6. Publish TwistStamped
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = "base_link"
            
            twist_msg.twist.linear.x = vel_x
            twist_msg.twist.linear.y = vel_y
            twist_msg.twist.linear.z = vel_z
            
            self.pub_twist.publish(twist_msg)
            return False

        except Exception as e:
            # self.get_logger().warn(f"TF Error: {e}")
            return False

    def stop_robot(self):
        stop_twist = TwistStamped()
        stop_twist.header.stamp = self.get_clock().now().to_msg()
        self.pub_twist.publish(stop_twist)
        
        stop_joints = JointJog()
        stop_joints.header.stamp = self.get_clock().now().to_msg()
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

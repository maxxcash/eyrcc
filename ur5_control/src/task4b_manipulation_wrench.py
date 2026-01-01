#!/usr/bin/env python3
"""
Arm Full Sequence Node (Real Hardware Optimized)
------------------------------------------------
1. Scans for ArUco marker -> Picks -> Places in Bin.
2. Scans for Bad Fruits -> Picks -> Places in Disposal.
3. Uses /tcp_pose_raw for feedback (Real Hardware Feedback).
4. Uses TwistStamped with Frame ID for explicit safety.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray, Float32
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from rclpy.duration import Duration
import numpy as np
import rclpy.time

class ArmFullSequence(Node):
    def __init__(self):
        super().__init__('arm_full_sequence_node') 

        # ------------------ TF (Target Detection) -------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------ Publishers & Services -------------------
        # HARDWARE NOTE: Using TwistStamped is mandatory for MoveIt Servo
        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.magnet_client = self.create_client(SetBool, '/magnet')

        # ------------------ Feedback Subscribers -------------------
        # 1. Robot Pose (TCP) - Using external topic instead of TF for feedback
        self.current_pos = None
        self.current_quat = None
        self.tcp_sub = self.create_subscription(
            Float64MultiArray, 
            '/tcp_pose_raw', 
            self.tcp_callback, 
            10
        )

        # 2. Force Sensor (Grasp Check)
        self.current_force_z = 0.0
        self.GRASP_THRESHOLD = 10.0 
        self.force_sub = self.create_subscription(
            Float32, 
            '/net_wrench', 
            self.force_callback, 
            10
        )

        # ------------------ Startup Wait -------------------
        self.get_logger().info(" HARDWARE CHECK: Waiting for Magnet Service...")
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info(" HARDWARE CHECK: Magnet Service Connected.")

        # ------------------ Timers -------------------
        self.control_timer = self.create_timer(0.02, self.control_loop) # 50 Hz Control
        self.scan_timer = self.create_timer(0.5, self.scan_tf_frames)   # 2 Hz Scanning

        # ------------------ Config -------------------
        self.base_frame = 'base_link'
        
        # HARDWARE NOTE: Speeds are conservative. Increase cautiously.
        self.max_linear_speed = 0.015  # 1.5 cm/s
        self.max_angular_speed = 0.15  # ~8 deg/s
        self.position_tolerance = 0.01
        self.orientation_tolerance = 0.1

        # ------------------ State Machine -------------------
        self.waypoints = []
        self.current_idx = 0
        
        # Detection Flags
        self.aruco_found = False
        self.detected_bad_fruits = set()
        self.bad_fruits_planned = False
        
        self.current_mode = "SCANNING" 

        # ------------------ HARDCODED POSES -------------------
        # 1. Bad Fruit Routine Poses
        self.badfruit_start = (np.array([0.109, 0.323, 0.5]),
                               np.array([0.005, 0.999, -0.011, 0.000]))
        
        self.badfruit_intermidiate = (np.array([-0.3821, 0.34222, 0.46803]),
                                      np.array([0.71306, -0.69933, -0.048928, -0.0087971]))
        
        self.drop_pose = (np.array([-0.85, 0.010, 0.25]),
                          np.array([-0.684, 0.726, 0.050, 0.008]))
        
        self.after_drop1 = (np.array([-0.45596, 0.002034, 0.23682]),
                            np.array([ 0.78317, -0.6218, -0.0026204, 0.0023389]))
        
        self.after_drop2 = (np.array([-0.45978, 0.0011078, 0.42318]),
                            np.array([0.78311, -0.62177, -0.0078359, 0.0090279]))

        # 2. ArUco Routine Poses
        self.aruco_start = (np.array([-0.094976, -0.24118, 0.56073]),
                            np.array([-0.040722, -0.6963, 0.71155, -0.084885]))
        
        self.aruco_after_pick1 = (np.array([-0.094976, -0.24118, 0.56073]),
                                  np.array([-0.040722, -0.6963, 0.71155, -0.084885]))
        
        self.aruco_after_pick2 = (np.array([0.25134, -0.063368, 0.56073]),
                                  np.array([-0.49599, 0.49039, -0.40614, 0.59039]))
        
        self.aruco_after_pick3 = (np.array([0.34344, -0.045919, 0.42938]),
                                  np.array([0.71063, -0.70352, 0.0041161, -0.0070831]))
        
        self.aruco_drop = (np.array([0.845, 0.024789, 0.33]),
                           np.array([0.71064, -0.70353, -0.0040817, 0.004818]))
        
        self.aruco_after_drop = (np.array([0.441, -0.055, 0.380]),
                                 np.array([0.666, 0.746, -0.007, -0.006]))

        self.get_logger().info(" NODE STARTED: Ready to scan.")

    # ------------------ CALLBACKS -------------------
    def force_callback(self, msg: Float32):
        self.current_force_z = msg.data

    def tcp_callback(self, msg):
        """Parse /tcp_pose_raw [x, y, z, rx, ry, rz]"""
        if len(msg.data) < 6:
            return
        
        # Position
        self.current_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])
        
        # Orientation (Euler [rad] -> Quat)
        try:
            r = R.from_euler('xyz', [msg.data[3], msg.data[4], msg.data[5]], degrees=False)
            self.current_quat = r.as_quat()
        except Exception:
            pass

    # ------------------ SEQUENCE BUILDING -------------------
    def scan_tf_frames(self):
        """Scans TF tree to find targets and build the plan dynamically."""
        try:
            frames = self.tf_buffer.all_frames_as_string()
        except Exception:
            return

        # 1. ARUCO DETECTION
        if (not self.aruco_found) and ('1425_fertilizer_1' in frames):
            try:
                # HARDWARE NOTE: rclpy.time.Time() = Time 0. This gets the LATEST transform.
                # Do NOT use self.get_clock().now() here, it will fail on real hardware due to delays.
                trans = self.tf_buffer.lookup_transform(
                    self.base_frame, '1425_fertilizer_1', rclpy.time.Time(), timeout=Duration(seconds=0.5))
                
                self.aruco_found = True
                self.get_logger().info(" TARGET ACQUIRED: ArUco Marker.")
                self.build_aruco_sequence(trans)
                self.current_mode = "EXECUTING_ARUCO"
            except Exception as e:
                self.get_logger().warn(f" ArUco lookup failed: {e}")

        # 2. BAD FRUIT DETECTION
        if self.current_mode in ["EXECUTING_ARUCO", "BAD_FRUIT"]:
            for line in frames.splitlines():
                if '1425_bad_fruit_' not in line: continue
                
                # Extract frame name
                frame = [p for p in line.strip().split() if '1425_bad_fruit_' in p][0]
                
                if frame not in self.detected_bad_fruits:
                    try:
                        # HARDWARE NOTE: Using Time 0 (latest available)
                        self.tf_buffer.lookup_transform(
                            self.base_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.5))
                        self.detected_bad_fruits.add(frame)
                        self.get_logger().info(f" TARGET ACQUIRED: {frame} ({len(self.detected_bad_fruits)}/3)")
                    except Exception:
                        pass

            # 3. TRIGGER BAD FRUIT PLAN (Only after 3 found & ArUco sequence underway/done)
            if len(self.detected_bad_fruits) >= 3 and not self.bad_fruits_planned:
                # We append these to the END of the existing waypoint list
                self.get_logger().info(" PLANNING: 3 Bad Fruits found. Appending to sequence.")
                self.build_badfruit_sequences()
                self.bad_fruits_planned = True

    def build_aruco_sequence(self, obj_tf):
        # Extract ArUco Pose
        pos_obj = np.array([obj_tf.transform.translation.x, obj_tf.transform.translation.y, obj_tf.transform.translation.z])
        quat_obj = np.array([obj_tf.transform.rotation.x, obj_tf.transform.rotation.y, obj_tf.transform.rotation.z, obj_tf.transform.rotation.w])

        # --- ARUCO PATH ---
        self.waypoints.append((self.aruco_start[0], self.aruco_start[1], 'none'))
        self.waypoints.append((pos_obj, quat_obj, 'magnet_on'))  # PICK
        self.waypoints.append((self.aruco_after_pick1[0], self.aruco_after_pick1[1], 'none'))
        self.waypoints.append((self.aruco_after_pick2[0], self.aruco_after_pick2[1], 'none'))
        self.waypoints.append((self.aruco_after_pick3[0], self.aruco_after_pick3[1], 'none'))
        self.waypoints.append((self.aruco_drop[0], self.aruco_drop[1], 'magnet_off')) # DROP
        self.waypoints.append((self.aruco_after_drop[0], self.aruco_after_drop[1], 'none'))
        self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none')) # Move to wait pose

    def build_badfruit_sequences(self):
        sorted_fruits = sorted(list(self.detected_bad_fruits))
        
        for f in sorted_fruits:
            try:
                # HARDWARE NOTE: Time 0 (latest)
                trans = self.tf_buffer.lookup_transform(self.base_frame, f, rclpy.time.Time())
                pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
                quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])

                z_up = pos.copy()
                z_up[2] += 0.09 # Hover above

                # --- FRUIT PATH (Per Fruit) ---
                self.waypoints.append((pos, quat, 'magnet_on')) # PICK
                self.waypoints.append((z_up, quat, 'none'))
                self.waypoints.append((self.badfruit_intermidiate[0], self.badfruit_intermidiate[1], 'none'))
                self.waypoints.append((self.drop_pose[0], self.drop_pose[1], 'magnet_off')) # DROP
                self.waypoints.append((self.after_drop1[0], self.after_drop1[1], 'none'))
                self.waypoints.append((self.after_drop2[0], self.after_drop2[1], 'none'))
                self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none')) # Return to start
            except Exception:
                pass

    # ------------------ SAFETY: EXPLICIT ZERO VELOCITY -------------------
    def stop_arm(self):
        """
        Publishes explicit 0 velocity to 'base_link'.
        Crucial for real hardware to prevent drift or latching.
        """
        stop_cmd = TwistStamped()
        stop_cmd.header.stamp = self.get_clock().now().to_msg()
        stop_cmd.header.frame_id = self.base_frame # Robot won't stop unless frame matches!
        # Explicitly set zeros (though default is 0, this is clearer)
        stop_cmd.twist.linear.x = 0.0; stop_cmd.twist.linear.y = 0.0; stop_cmd.twist.linear.z = 0.0
        stop_cmd.twist.angular.x = 0.0; stop_cmd.twist.angular.y = 0.0; stop_cmd.twist.angular.z = 0.0
        self.twist_pub.publish(stop_cmd)

    def control_magnet(self, state: bool):
        req = SetBool.Request()
        req.data = state
        self.magnet_client.call_async(req)

    def control_loop(self):
        # 1. Wait for Valid Feedback (Hardware Check)
        if self.current_pos is None or self.current_quat is None:
            return 

        # 2. Wait for Plan (Safety Idle)
        if not self.waypoints:
            self.stop_arm()
            return

        # 3. Check Mission Complete (Safety Stop)
        if self.current_idx >= len(self.waypoints):
            self.stop_arm()
            if self.current_mode != "COMPLETED":
                self.get_logger().info(" MISSION COMPLETE.")
                self.current_mode = "COMPLETED"
            return

        # 4. Control Logic
        tgt_pos, tgt_quat, action = self.waypoints[self.current_idx]

        # P-Controller
        pos_err = tgt_pos - self.current_pos
        kp_pos = 0.5
        linear_cmd = kp_pos * pos_err

        current_rot = R.from_quat(self.current_quat)
        target_rot = R.from_quat(tgt_quat)
        rot_err = target_rot * current_rot.inv()
        rotvec = rot_err.as_rotvec()
        kp_ang = 4.0
        angular_cmd = kp_ang * rotvec

        # Speed Clipping (Safety Limit)
        if np.linalg.norm(linear_cmd) > self.max_linear_speed:
            linear_cmd *= (self.max_linear_speed / np.linalg.norm(linear_cmd))
        if np.linalg.norm(angular_cmd) > self.max_angular_speed:
            angular_cmd *= (self.max_angular_speed / np.linalg.norm(angular_cmd))

        # Publish TwistStamped
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = float(linear_cmd[0]), float(linear_cmd[1]), float(linear_cmd[2])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = float(angular_cmd[0]), float(angular_cmd[1]), float(angular_cmd[2])
        self.twist_pub.publish(msg)

        # 5. Check Arrival & Execute Action
        pos_dist = np.linalg.norm(pos_err)
        angle_dist = np.linalg.norm(rotvec)

        if pos_dist < self.position_tolerance and angle_dist < self.orientation_tolerance:
            
            # Action: PICK
            if action == 'magnet_on':
                self.control_magnet(True)
                # Force check - SAFETY STOP IF NO GRASP
                if self.current_force_z > self.GRASP_THRESHOLD:
                    self.get_logger().info(f" GRASPED (Force: {self.current_force_z:.1f}). Next.")
                    self.current_idx += 1
                else:
                    self.stop_arm() # Explicitly Halt while waiting
                    self.get_logger().warn(" Waiting for grasp...", throttle_duration_sec=2)
            
            # Action: DROP
            elif action == 'magnet_off':
                self.stop_arm() # Stabilize before drop
                self.control_magnet(False)
                self.get_logger().info(" DROPPED object.")
                self.current_idx += 1
            
            # Action: MOVE
            else:
                self.current_idx += 1
                self.get_logger().info(f" Waypoint {self.current_idx}/{len(self.waypoints)} reached.")

def main(args=None):
    rclpy.init(args=args)
    node = ArmFullSequence()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # HARDWARE NOTE: Emergency Stop on Exit
        node.get_logger().info(" SHUTDOWN: Stopping Arm...")
        node.stop_arm()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

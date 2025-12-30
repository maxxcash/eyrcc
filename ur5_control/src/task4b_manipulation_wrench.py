#!/usr/bin/env python3
"""
Arm full pick-and-place node with EXPLICIT ZERO VELOCITY SAFETY.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from rclpy.duration import Duration
from std_srvs.srv import SetBool
import numpy as np
import rclpy.time
from std_msgs.msg import Float32


class ArmFullSequence(Node):
    def __init__(self):
        super().__init__('arm_full_sequence_node') 

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # publishers
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # magnet service
        self.magnet_client = self.create_client(SetBool, '/magnet')

        # ------------------ Force Monitoring -------------------
        self.current_force_z = 0.0
        self.force_sub = self.create_subscription(
            Float32,
            '/net_wrench',
            self.force_callback,
            10
        )
        # Threshold to consider an object "grasped"
        self.GRASP_THRESHOLD = 10.0 

        self.get_logger().info(" SYSTEM STARTUP: Waiting for Magnet Service...")
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info(" SYSTEM READY: Magnet Service Connected.")

        # timers
        self.control_timer = self.create_timer(0.02, self.control_loop)   # 50 Hz
        self.scan_timer = self.create_timer(0.5, self.scan_tf_frames)     # scan TFs

        # frames
        self.base_frame = 'base_link'
        self.ee_frame = 'ee_link'

        # controller params
        self.max_linear_speed = 0.01
        self.max_angular_speed = 0.1
        self.position_tolerance = 0.01
        self.orientation_tolerance = 0.1

        # sequence bookkeeping
        self.waypoints = []
        self.current_idx = 0
        self.detected_bad_fruits = set()
        self.aruco_found = False
        self.bad_fruits_planned = False

        # state tracking
        self.current_mode = "IDLE" 

        # static/known poses
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

        # ArUco sequence poses 
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
 
        self.get_logger().info(" NODE STARTED: Scanning for targets...")

    # ------------------ SAFETY HELPER -------------------
    def stop_arm(self):
        """Publishes explicit zero velocity to halt the robot."""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0; stop_cmd.linear.y = 0.0; stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0; stop_cmd.angular.y = 0.0; stop_cmd.angular.z = 0.0
        self.twist_pub.publish(stop_cmd)

    # ------------------ TF scanning -------------------
    def scan_tf_frames(self):
        try:
            frames = self.tf_buffer.all_frames_as_string()
        except Exception:
            return

        # 1. Look for ArUco
        if (not self.aruco_found) and ('1425_fertilizer_1' in frames):
            try:
                trans = self.tf_buffer.lookup_transform(self.base_frame, '1425_fertilizer_1', rclpy.time.Time(), timeout=Duration(seconds=0.5))
                self.aruco_found = True
                self.get_logger().info(" VISUAL CONTACT: ArUco Marker '1425_fertilizer_1' Found.")
                self.build_aruco_sequence(trans)
                self.current_mode = "ARUCO"
            except Exception as e:
                self.get_logger().warn(f" ArUco lookup failed: {e}")

        # 2. Look for Bad Fruits
        for line in frames.splitlines():
            if '1425_bad_fruit_' not in line:
                continue
            frame = [p for p in line.strip().split() if '1425_bad_fruit_' in p][0]
            if frame in self.detected_bad_fruits:
                continue
            try:
                self.tf_buffer.lookup_transform(self.base_frame, frame, rclpy.time.Time(), timeout=Duration(seconds=0.5))
                self.detected_bad_fruits.add(frame)
                self.get_logger().info(f" VISUAL CONTACT: Bad Fruit '{frame}' Found ({len(self.detected_bad_fruits)}/3)")
            except Exception:
                pass

        # 3. Plan Bad Fruits
        if self.aruco_found and len(self.detected_bad_fruits) >= 3 and not self.bad_fruits_planned:
            if not any(np.allclose(wp[0], self.drop_pose[0], atol=1e-6) for wp in self.waypoints):
                self.get_logger().info(" PLANNING: All targets found. Generating Bad Fruit Sequence.")
                self.build_badfruit_sequences()
                self.bad_fruits_planned = True

    # ------------------ Build sequences -------------------
    def build_aruco_sequence(self, obj3_transform):
        self.waypoints = []
        self.current_idx = 0

        pos_obj3 = np.array([obj3_transform.transform.translation.x,
                             obj3_transform.transform.translation.y ,
                             obj3_transform.transform.translation.z], dtype=float)
        quat_obj3 = np.array([obj3_transform.transform.rotation.x,
                              obj3_transform.transform.rotation.y,
                              obj3_transform.transform.rotation.z,
                              obj3_transform.transform.rotation.w], dtype=float)

        #aruco poses to drop on ebot
        self.waypoints.append((self.aruco_start[0], self.aruco_start[1], 'none'))
        self.waypoints.append((pos_obj3, quat_obj3, 'magnet_on'))
        self.waypoints.append((self.aruco_after_pick1[0], self.aruco_after_pick1[1], 'none'))
        self.waypoints.append((self.aruco_after_pick2[0], self.aruco_after_pick2[1], 'none'))
        self.waypoints.append((self.aruco_after_pick3[0], self.aruco_after_pick3[1], 'none'))
        self.waypoints.append((self.aruco_drop[0], self.aruco_drop[1], 'magnet_off'))
        self.waypoints.append((self.aruco_after_drop[0], self.aruco_after_drop[1], 'none'))
        self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none'))

        #aruco poses to drop in the bin
        # self.waypoints.append((self.aruco_start[0], self.aruco_start[1], 'none'))
        # self.waypoints.append((pos_obj3, quat_obj3, 'magnet_on'))
        # self.waypoints.append((self.aruco_after_pick1[0], self.aruco_after_pick1[1], 'none'))
        # self.waypoints.append((self.aruco_after_pick2[0], self.aruco_after_pick2[1], 'none'))
        # self.waypoints.append((self.aruco_after_drop[0], self.aruco_after_drop[1], 'none'))
        # self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none'))
        # self.waypoints.append((self.badfruit_intermidiate[0], self.badfruit_intermidiate[1], 'none'))
        # self.waypoints.append((self.drop_pose[0], self.drop_pose[1], 'magnet_off'))
        # self.waypoints.append((self.after_drop1[0], self.after_drop1[1], 'none'))
        # self.waypoints.append((self.after_drop2[0], self.after_drop2[1], 'none'))
        # self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none'))
     
        
        self.get_logger().info(f" PLANNING: ArUco Sequence Generated ({len(self.waypoints)} waypoints). Executing...")

    def build_badfruit_sequences(self):
        ordered = sorted(list(self.detected_bad_fruits))
        for f in ordered:
            try:
                trans = self.tf_buffer.lookup_transform(self.base_frame, f, rclpy.time.Time(), timeout=Duration(seconds=0.5))
                pos = np.array([trans.transform.translation.x,
                                trans.transform.translation.y,
                                trans.transform.translation.z ], dtype=float)
                quat = np.array([trans.transform.rotation.x,
                                 trans.transform.rotation.y,
                                 trans.transform.rotation.z,
                                 trans.transform.rotation.w], dtype=float)

                z_down = pos.copy()
                z_down[2] = z_down[2] + 0.09   # go up by 5cm to pick

                self.waypoints.append((pos, quat, 'magnet_on'))
                self.waypoints.append((z_down, quat, 'none'))
                self.waypoints.append((self.badfruit_intermidiate[0], self.badfruit_intermidiate[1], 'none'))
                self.waypoints.append((self.drop_pose[0], self.drop_pose[1], 'magnet_off'))
                self.waypoints.append((self.after_drop1[0], self.after_drop1[1], 'none'))
                self.waypoints.append((self.after_drop2[0], self.after_drop2[1], 'none'))
                self.waypoints.append((self.badfruit_start[0], self.badfruit_start[1], 'none'))
            except Exception:
                pass
        self.get_logger().info(f" PLANNING: Bad Fruit Sequence Generated. Appending to queue.")

    # ------------------ Control loop -------------------
    def control_loop(self):
        if not self.waypoints:
            self.stop_arm() # Safe idle
            return

        if self.current_idx >= len(self.waypoints):
            self.stop_arm() # EXPLICIT STOP AT END
            if self.current_mode != "COMPLETED":
                self.get_logger().info(" MISSION ACCOMPLISHED: All sequences finished.")
                self.current_mode = "COMPLETED"
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            current_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=float)
            current_quat = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ], dtype=float)
        except Exception:
            self.stop_arm() # Stop if TF fails
            return

        tgt_pos, tgt_quat, action = self.waypoints[self.current_idx]

        # ----------- P-Control -----------
        kp_pos = 0.5
        kp_ang = 4.0

        pos_err = tgt_pos - current_pos
        linear_cmd = kp_pos * pos_err

        current_rot = R.from_quat(current_quat)
        target_rot = R.from_quat(tgt_quat)
        rot_err = target_rot * current_rot.inv()
        rotvec = rot_err.as_rotvec()
        angular_cmd = kp_ang * rotvec

        # Clip speeds
        lin_norm = np.linalg.norm(linear_cmd)
        if lin_norm > self.max_linear_speed:
            linear_cmd = linear_cmd * (self.max_linear_speed / lin_norm)
        ang_norm = np.linalg.norm(angular_cmd)
        if ang_norm > self.max_angular_speed:
            angular_cmd = angular_cmd * (self.max_angular_speed / ang_norm)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = linear_cmd
        twist.angular.x, twist.angular.y, twist.angular.z = angular_cmd
        self.twist_pub.publish(twist)

        # ----------- Check if reached -----------
        pos_dist = np.linalg.norm(pos_err)
        angle = np.linalg.norm(rotvec)
        
        if pos_dist < self.position_tolerance and angle < self.orientation_tolerance:
            # We are at the target pose. Now handle Action Logic with Force Check.
            
            # --- ACTION: MAGNET ON (PICK UP) ---
            if action == 'magnet_on':
                self.control_magnet(True)
                
                # Check Force Sensor
                if self.current_force_z > self.GRASP_THRESHOLD:
                    self.get_logger().info(f" GRASP SUCCESS (Force: {self.current_force_z:.1f} > {self.GRASP_THRESHOLD}). Moving Next.")
                    self.current_idx += 1
                else:
                    # Force is too low -> EXPLICITLY STOP and Wait
                    self.stop_arm()  # <--- SAFETY: FORCE STOP WHILE WAITING
                    self.get_logger().warn(f" Waiting for grasp... Current Force: {self.current_force_z:.1f} / {self.GRASP_THRESHOLD}", throttle_duration_sec=2.0)
                    # NOTE: We do NOT increment current_idx here, so we stay in this loop.

            # --- ACTION: MAGNET OFF (DROP) ---
            elif action == 'magnet_off':
                self.stop_arm() # Stop briefly to ensure clean drop
                self.control_magnet(False)
                self.get_logger().info(f" RELEASE COMPLETE. Moving Next.")
                self.current_idx += 1
            
            # --- ACTION: MOVE ONLY ---
            else:
                self.get_logger().info(f" Step {self.current_idx+1}/{len(self.waypoints)} Reached.")
                self.current_idx += 1

            # Detect mode switch (ArUco -> Bad Fruits)
            if self.current_mode == "ARUCO" and np.allclose(tgt_pos, self.badfruit_start[0], atol=0.02):
                self.current_mode = "BAD_FRUIT"
                self.get_logger().info(" SWITCHING MODE: ArUco -> Bad Fruits")

    def control_magnet(self, state: bool):
        req = SetBool.Request()
        req.data = state
        self.magnet_client.call_async(req)

    def force_callback(self, msg: Float32):
        self.current_force_z = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = ArmFullSequence()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(" EMERGENCY STOP: Sending zero velocity command.")
        node.stop_arm()  # <--- SAFETY: STOP ON EXIT
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

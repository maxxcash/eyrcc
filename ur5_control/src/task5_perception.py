#!/usr/bin/env python3
'''
Team ID: 1425
Theme: Krishi Cobot
File: task3b_perception.py
'''
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import cv2
import numpy as np
import cv2 
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

# ROS Utilities
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from tf_transformations import quaternion_from_euler
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
import numpy as np
import cv2
import cv2.aruco as aruco
from tf_transformations import quaternion_from_euler
import rclpy.time
from typing import Tuple
from scipy.spatial.transform import Rotation as R

class Detection(Node):
    def __init__(self):
        super().__init__('image_depth_subscriber')
        
        # --- CONFIGURATION ---
        self.team_id = '1425'  # Update your Team ID here
        self.max_fruits = 3
        self.persistence_threshold = 10.0
        self.distance_threshold = 0.1
        self.timeout_threshold = 500.0

        # --- SETUP ---
        self.bridge = CvBridge()   
        self.cv_image = None       
        self.depth_image = None    

        # Intrinsic Defaults
        self.fx, self.fy = 915.30, 914.03   
        self.cx, self.cy = 642.72, 361.97   

        # TF Objects
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, '/camera/camera/camera_info', self.camera_info_callback, 10)

        # Tracking State
        self.tracked_fruits = {}   
        self.next_fruit_id = 1     
        
        self.get_logger().info("Fruit Detection Node initialized.")

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.depth_image is not None:
                self.bad_fruit_detection(self.cv_image, self.depth_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error (RGB): {e}")

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error (Depth): {e}")

    def bad_fruit_detection(self, rgb_image, depth_image):
        output = rgb_image.copy()
        current_time = self.get_clock().now().nanoseconds / 1e9  

        # ROI Definition
        x1, y1, w1, h1 = 0, 180, 350, 450
        detection_region = output[y1:y1 + h1, x1:x1 + w1]
        hsv = cv2.cvtColor(detection_region, cv2.COLOR_BGR2HSV)

        # Thresholds
        lower_grey, upper_grey = np.array([0, 0, 80]), np.array([180, 40, 160])
        lower_green, upper_green = np.array([36, 25, 25]), np.array([86, 255, 255])

        grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        grey_contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grey_boxes = [cv2.boundingRect(c) for c in grey_contours]
        
        # 1. Detect Potential Fruits
        current_detections = []

        for c in green_contours:
            if cv2.contourArea(c) < 500: continue 

            x, y, w, h = cv2.boundingRect(c)
            # Only consider if overlapping with grey (bad fruit characteristic)
            if not any(self.is_overlapping((x, y, w, h), gb) for gb in grey_boxes):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0: continue
            
            # Global coordinates
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            full_cx, full_cy = x1 + cx, y1 + cy

            # Boundary checks
            h_img, w_img = depth_image.shape[:2]
            if not (0 <= full_cx < w_img and 0 <= full_cy < h_img): continue

            # Depth Calculation
            patch = depth_image[max(full_cy - 2, 0):min(full_cy + 3, h_img),
                                max(full_cx - 2, 0):min(full_cx + 3, w_img)].astype(np.float64)
            valid_mask = patch > 0
            if not np.any(valid_mask): continue

            depth_raw = np.median(patch[valid_mask])
            depth_m = depth_raw / 1000.0 if depth_image.dtype == np.uint16 else float(depth_raw)

            # 3D Reconstruction
            X = (full_cx - self.cx) * depth_m / self.fx
            Y = (full_cy - self.cy) * depth_m / self.fy
            Z = depth_m

            current_detections.append({
                "center": (full_cx, full_cy),
                "position_3d": (Z, -X, -Y),
                "width": w
            })

        # 2. Tracking Logic
        for detection in current_detections:
            cx, cy = detection["center"]
            best_match_id = None
            min_dist = float('inf')

            for fid, info in self.tracked_fruits.items():
                prev_cx, prev_cy = info['centroid']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if dist < self.distance_threshold and dist < min_dist:
                    min_dist = dist
                    best_match_id = fid

            if best_match_id is not None:
                self.tracked_fruits[best_match_id]['last_seen'] = current_time
                self.tracked_fruits[best_match_id]['centroid'] = (cx, cy)
                self.tracked_fruits[best_match_id]['data'] = detection
            elif len(self.tracked_fruits) < self.max_fruits:
                self.tracked_fruits[self.next_fruit_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'centroid': (cx, cy),
                    'data': detection
                }
                self.next_fruit_id += 1

        # 3. Publish / Cleanup
        ids_to_remove = []
        for fid, info in self.tracked_fruits.items():
            if current_time - info['last_seen'] > self.timeout_threshold:
                ids_to_remove.append(fid)
                continue

            duration_visible = current_time - info['first_seen']
            cx, cy = info['centroid']
            w = info['data']['width']

            if duration_visible > self.persistence_threshold:
                # Visualization
                cv2.rectangle(output, (cx - w//2, cy - w//2), (cx + w//2, cy + w//2), (0, 255, 0), 2)
                cv2.putText(output, f"BF_{fid} [OK]", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Publish TF
                fruit_info = info['data']
                fruit_info['id'] = fid
                self.publish_tf(fruit_info)
                self.republish_badfruit_in_base(fruit_info['id'])
            else:
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        for fid in ids_to_remove:
            del self.tracked_fruits[fid]

        # cv2.imshow("Detected Bad Fruits", output)
        return []

    def is_overlapping(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (x1 < x2 + w2 and x1 + w1 > x2) and (y1 < y2 + h2 and y1 + h1 > y2)
    
    def publish_tf(self, fruit_info):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f"cam{fruit_info['id']}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = fruit_info['position_3d']
        
        # Static rotation for fruit
        qx, qy, qz, qw = quaternion_from_euler(1.571, 0.0, 1.571)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)
        self.tf_broadcaster.sendTransform(t)

    def republish_badfruit_in_base(self, fruit_id):
        try:
            trans = self.tf_buffer.lookup_transform("base_link", f"cam{fruit_id}", rclpy.time.Time(), timeout=Duration(seconds=0.5))
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
            t.transform.translation = trans.transform.translation
            
            qx, qy, qz, qw = quaternion_from_euler(3.14, 0.0, -1.57)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")

# =========================================================
# ================= ARUCO TF NODE =========================
# =========================================================
class ArucoTF(Node):

    def __init__(self):
        super().__init__('aruco_tf_publisher')

        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # Camera intrinsics
        self.fx, self.fy = 915.30, 914.03
        self.cx, self.cy = 642.72, 361.97

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscriptions
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_cb, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)

        # ---------------- ARUCO SETUP (API SAFE) ----------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        if hasattr(cv2.aruco, "ArucoDetector"):
            # New OpenCV (>= 4.7)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params
            )
            self.use_new_api = True
            self.get_logger().info("Using NEW OpenCV ArUco API")
        else:
            # Old OpenCV (<= 4.6)  ← eYRC default
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
            self.get_logger().info("Using OLD OpenCV ArUco API")

    def color_cb(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.depth_image is not None:
            self.aruco_detection(self.cv_image, self.depth_image)

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    # ---------------------------------------------------------
    # ---------------- ARUCO DETECTION ------------------------
    # ---------------------------------------------------------
    def aruco_detection(self, rgb_image, depth_image):

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Detect markers (API SAFE)
        if self.use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

        if ids is None:
            return

        cam_mat = np.array([
            [915.30, 0.0, 642.72],
            [0.0, 914.03, 361.97],
            [0.0, 0.0, 1.0]
        ])
        dist_mat = np.zeros((5, 1))
        marker_size = 0.13  # meters

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, cam_mat, dist_mat
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            r = R.from_rotvec(rvec.flatten())
            euler_angles = r.as_euler('xyz', degrees=True)
            pitch = euler_angles[0]
            is_flat = 60 < abs(pitch) < 90
                    

            # -------- STANDARD YAW COMPUTATION --------
            R_cam, _ = cv2.Rodrigues(rvec)
            yaw_rad = np.arctan2(R_cam[1, 0], R_cam[0, 0])
            yaw_deg = np.degrees(yaw_rad)

            # Camera frame position
            X = tvec[0][0]
            Y = tvec[0][1]
            Z = tvec[0][2]

            aruco_data = {
                "id": int(marker_id),
                "position": (Z, -X, -Y),
                "yaw": yaw_deg,
                "is_flat": is_flat
            }

            self.publish_aruco_tf(aruco_data)
            self.republish_aruco_in_base(aruco_data)

    # ---------------------------------------------------------
    # ---------------- TF PUBLISHERS --------------------------
    # ---------------------------------------------------------
    def publish_aruco_tf(self, aruco_info):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "camera_link"
        t.child_frame_id = f"camera_{aruco_info['id']}"

        t.transform.translation.x = aruco_info["position"][0]
        t.transform.translation.y = aruco_info["position"][1]
        t.transform.translation.z = aruco_info["position"][2]

        # Use REAL yaw
        q = quaternion_from_euler(0.0, 0.0, np.radians(aruco_info["yaw"]))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def republish_aruco_in_base(self, aruco_info, teamid='1425'):
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                f"camera_{aruco_info['id']}",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = f"{teamid}_fertilizer_1"

            t.transform.translation.x = trans.transform.translation.x + 0
            t.transform.translation.y = trans.transform.translation.y + 0
            t.transform.translation.z = trans.transform.translation.z + 0

            # Fixed orientation for UR5 grasp (as per eYRC)
            if aruco_info['is_flat']:
                qx, qy, qz, qw = quaternion_from_euler(1.571, 3.14, 0.0) 
            else:
                qx, qy, qz, qw = quaternion_from_euler(3.14, 0.0, -1.57)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed for ArUco {aruco_info['id']}: {e}"
            )


# =========================================================
# ================= MAIN =================================
# =========================================================
def main(args=None):
    rclpy.init(args=args)

    aruco_node = ArucoTF()
    detection_node = Detection()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(aruco_node)
    executor.add_node(detection_node)


    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detection_node.destroy_node()
        aruco_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

'''
# Team ID: <Team-ID>
# Theme: Krishi c0b0t
# Author List: <Your Name(s)>
# Filename: image_depth_subscriber.py
# Functions: __init__, camera_info_callback, image_callback, depth_callback,
#            bad_fruit_detection, is_overlapping, publish_tf, republish_badfruit_in_base, main
# Global variables: None
'''

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.duration import Duration
from tf_transformations import quaternion_from_euler
import rclpy
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
import cv2.aruco as aruco
from typing import List, Tuple
from tf_transformations import quaternion_from_euler
from rclpy.duration import Duration
import rclpy.time



class Detection(Node):
    '''
    Purpose:
        A ROS2 node that subscribes to RGB, depth, and camera info topics,
        detects "bad fruits" in the image, estimates their 3D position,
        tracks them over time, and publishes corresponding transforms 
        only after they persist for a set duration.
    '''
 
    def __init__(self):
        super().__init__('image_depth_subscriber')

        # Initialize CvBridge for ROS <-> OpenCV conversions
        self.bridge = CvBridge()   
        self.cv_image = None       
        self.depth_image = None    

        # Camera intrinsics (defaults until CameraInfo is received)
        self.fx, self.fy = 915.30, 914.03   
        self.cx, self.cy = 642.72, 361.97   

        # TF buffer, listener, broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Topics
        self.rgb_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.caminfo_topic = '/camera/camera/camera_info'

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.caminfo_topic, self.camera_info_callback, 10)

        # --- TRACKING VARIABLES ---
        # Stores fruit history: {id: {'first_seen': time, 'last_seen': time, 'centroid': (x,y), 'data': dict}}
        self.tracked_fruits = {}   
        self.next_fruit_id = 1     
        
        # CONFIGURATION
        self.persistence_threshold = 40.0  # Seconds a fruit must be seen before publishing
        self.distance_threshold = 50.0    # Pixel distance to consider it the "same" fruit
        self.timeout_threshold = 0.5      # Seconds before we delete a lost fruit

        self.get_logger().info("Subscribed to RGB, Depth, and Camera Info topics.")

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
            self.get_logger().error(f"CV Bridge Error in image_callback: {e}")

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error in depth_callback: {e}")

    def bad_fruit_detection(self, rgb_image, depth_image):
        '''
        Purpose:
            Detect bad fruits, track them over time, and publish TFs only if consistent.
        '''
        output = rgb_image.copy()
        current_time = self.get_clock().now().nanoseconds / 1e9  # Current time in Seconds

        # ROI for fruit detection
        x1, y1, w1, h1 = 0, 180, 350, 450
        detection_region = output[y1:y1 + h1, x1:x1 + w1]
        hsv = cv2.cvtColor(detection_region, cv2.COLOR_BGR2HSV)

        # Color thresholds
        lower_grey, upper_grey = np.array([0, 0, 80]), np.array([180, 40, 160])
        lower_green, upper_green = np.array([36, 25, 25]), np.array([86, 255, 255])

        grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        grey_contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        grey_boxes = [cv2.boundingRect(c) for c in grey_contours]
        
        # 1. Gather all raw detections in the current frame
        current_frame_detections = []

        for c in green_contours:
            if cv2.contourArea(c) < 500:
                continue 

            x, y, w, h = cv2.boundingRect(c)
            if not any(self.is_overlapping((x, y, w, h), gb) for gb in grey_boxes):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            full_cx, full_cy = x1 + cx, y1 + cy

            h_img, w_img = depth_image.shape[:2]
            if not (0 <= full_cx < w_img and 0 <= full_cy < h_img):
                continue

            patch = depth_image[max(full_cy - 2, 0):min(full_cy + 3, h_img),
                                max(full_cx - 2, 0):min(full_cx + 3, w_img)].astype(np.float64)
            valid_mask = patch > 0
            if not np.any(valid_mask):
                continue

            depth_raw = np.median(patch[valid_mask])
            depth_m = depth_raw / 1000.0 if depth_image.dtype == np.uint16 else float(depth_raw)

            X = (full_cx - self.cx) * depth_m / self.fx
            Y = (full_cy - self.cy) * depth_m / self.fy
            Z = depth_m

            detection_data = {
                "center": (full_cx, full_cy),
                "position_3d": (Z, -X, -Y),
                "width": w
            }
            current_frame_detections.append(detection_data)

        # 2. Match current detections to tracked fruits (Centroid Tracking)
        matched_ids = set()

        for detection in current_frame_detections:
            cx, cy = detection["center"]
            best_match_id = None
            min_dist = float('inf')

            # Find closest existing tracked fruit
            for fid, info in self.tracked_fruits.items():
                prev_cx, prev_cy = info['centroid']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if dist < self.distance_threshold:
                    if dist < min_dist:
                        min_dist = dist
                        best_match_id = fid

            # Update existing or create new
            if best_match_id is not None:
                self.tracked_fruits[best_match_id]['last_seen'] = current_time
                self.tracked_fruits[best_match_id]['centroid'] = (cx, cy)
                self.tracked_fruits[best_match_id]['data'] = detection 
                matched_ids.add(best_match_id)
            else:
                new_id = self.next_fruit_id
                self.tracked_fruits[new_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'centroid': (cx, cy),
                    'data': detection
                }
                self.next_fruit_id += 1
                matched_ids.add(new_id)

        # 3. Process Logic: Publish only if confirmed, remove if lost
        ids_to_remove = []
        
        for fid, info in self.tracked_fruits.items():
            # Remove fruits lost for too long
            if current_time - info['last_seen'] > self.timeout_threshold:
                ids_to_remove.append(fid)
                continue

            # Check duration
            duration_visible = current_time - info['first_seen']
            cx, cy = info['centroid']
            w = info['data']['width']

            if duration_visible > self.persistence_threshold:
                # CONFIRMED: Draw Green & Publish
                cv2.rectangle(output, (cx - w//2, cy - w//2), (cx + w//2, cy + w//2), (0, 255, 0), 2)
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(output, f"bad_fruit_{fid}", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                fruit_info = info['data']
                fruit_info['id'] = fid
                self.publish_tf(fruit_info)
                self.republish_badfruit_in_base(fruit_info['id'])
            else:
                # PENDING: Draw Red Dot & Wait
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(output, f"Verifying... {duration_visible:.1f}s", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Cleanup
        for fid in ids_to_remove:
            del self.tracked_fruits[fid]

        # cv2.imshow("Detected Bad Fruits", output)
        cv2.waitKey(1)
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
        qx, qy, qz, qw = quaternion_from_euler(1.571, 0.0, 1.571)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

        self.tf_broadcaster.sendTransform(t)

    def republish_badfruit_in_base(self, fruit_id, teamid='1425'):
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link",
                f"cam{fruit_id}",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = f'{teamid}_bad_fruit_{fruit_id}'
            t.transform.translation = trans.transform.translation

            qx, qy, qz, qw = quaternion_from_euler(3.14, 0.0, -1.57)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

            self.tf_broadcaster.sendTransform(t)
            # self.get_logger().info(f"Republished cam{fruit_id} in base_link")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")


class ArucoTF(Node):

    def __init__(self):
        super().__init__('aruco_tf_publisher')             
        self.color_cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                          # depth raw image variable (from depthimagecb())
        self.bridge = CvBridge()                                                       # OpenCV <-> ROS Image message converter


        self.fx, self.fy = 915.30, 914.03   # Focal lengths
        self.cx, self.cy = 642.72, 361.97   # Principal points

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)


    def colorimagecb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # cv2.imshow("RGB Image", self.cv_image)

            if self.depth_image is not None:
                self.aruco_detection(self.cv_image, self.depth_image)

            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error in image_callback: {e}")


    def depthimagecb(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype('uint8')

            # cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error in depth_callback: {e}")



    def aruco_detection(self, rgb_image, depth_image):
        aruco_area_threshold = 1500
        cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375],
                            [0.0, 914.0320434570312, 361.9780578613281],
                            [0.0, 0.0, 1.0]])
        dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        size_of_aruco_m = 0.13

        output = rgb_image.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        aruco_info = []

        if ids is not None:
            aruco.drawDetectedMarkers(output, corners, ids)
            for i, marker_corner in enumerate(corners):
                marker_id = ids[i][0]
                area, width = self.calculate_rectangle_area(marker_corner)

                if area > aruco_area_threshold:
                    # Center calculation
                    pts = marker_corner[0]
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    else:
                        center_x = int(np.mean(pts[:, 0]))
                        center_y = int(np.mean(pts[:, 1]))

                    # Depth at center (safe indexing + median filter)
                    y_start, y_end = max(0, center_y - 2), min(depth_image.shape[0], center_y + 3)
                    x_start, x_end = max(0, center_x - 2), min(depth_image.shape[1], center_x + 3)
                    depth_window = depth_image[y_start:y_end, x_start:x_end]
                    distance = float(np.median(depth_window).item())


                    # Pose estimation
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        marker_corner, size_of_aruco_m, cam_mat, dist_mat
                    )
                    rvec, tvec = rvec[0], tvec[0]

                    r = R.from_rotvec(rvec.flatten())
                    # Get Euler angles in degrees (Sequence: x=Pitch, y=Yaw, z=Roll)
                    # Note: 'xyz' sequence is usually standard for simple pitch checks
                    euler_angles = r.as_euler('xyz', degrees=True)
                    pitch = euler_angles[0]

                    # Logic: Vertical markers usually have pitch near 0 or 180 (facing camera).
                    # Flat markers (floor) usually have pitch near +/- 90.
                    # We use a threshold (e.g., > 45 degrees means it's tilting significantly)
                    is_flat = abs(pitch) > 60 and abs(pitch) < 90
                    
                    # Convert rotation to yaw angle
                    r = R.from_rotvec(rvec.flatten())
                    yaw_angle_deg = float(r.as_euler('zyx', degrees=True)[0])


                    X = (center_x - self.cx) * distance / self.fx
                    Y = (center_y - self.cy) * distance / self.fy
                    Z = distance

                    angle_aruco = (0.788*yaw_angle_deg) - ((yaw_angle_deg**2)/3160)



                    # Draw axis & annotations
                    cv2.drawFrameAxes(output, cam_mat, dist_mat, rvec, tvec, size_of_aruco_m * 0.5)
                    cv2.circle(output, (center_x, center_y), 5, (0, 255, 255), -1)
                    # cv2.putText(
                    #     output,
                    #     f"Dist:{float(distance):.2f}m Yaw:{float(yaw_angle_deg):.1f}",
                    #     (center_x, center_y),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     (0, 255, 0),
                    #     2
                    # )

                    aruco_data = {
                        "id": int(marker_id),
                        "position": (Z, -X, -Y),
                        "yaw": float(angle_aruco),
                        "is_flat": is_flat
                    }

                    aruco_info.append(aruco_data)
                    self.aruco_publish_tf(aruco_data)
                    self.republish_aruco_in_base(aruco_data)
            # cv2.imshow("Aruco Detection", output)
            cv2.waitKey(1)

        return aruco_info

    def calculate_rectangle_area(self, coordinates: np.ndarray) -> Tuple[float, float]:
        area = 0.0
        width = 0.0 
        corners = coordinates.reshape(4, 2)
        top_left = corners[0]
        top_right = corners[1]
        bottom_right = corners[2]
        bottom_left = corners[3]

        width = np.linalg.norm(top_right - top_left)
        height = np.linalg.norm(top_left - bottom_left)
        area = width * height

        return area, width



    def aruco_publish_tf(self, aruco_info, teamid='1425'):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f"camera_{aruco_info['id']}"

        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = aruco_info['position']
      
            # Marker is standing vertical (Your original working values)
        qx, qy, qz, qw = quaternion_from_euler(1.571, 2.355, 0.0)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

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
                t.transform.translation.x = trans.transform.translation.x + 0.0
                t.transform.translation.y = trans.transform.translation.y - 0.005
                t.transform.translation.z = trans.transform.translation.z + 0.0
                if aruco_info['is_flat']:
                    # Marker is lying flat on the ground.
                    # You likely need to rotate X by 90 degrees compared to your vertical setup.
                    # Try this (Standard Flat):
                    qx, qy, qz, qw = quaternion_from_euler(1.571, 3.14, 0.0) 
                    
                    # NOTE: If the above orientation isn't perfect, try adjusting the Pitch (middle value)
                    # e.g., quaternion_from_euler(1.571, -1.571, 1.571)
                else:
                    # Marker is horizontal 
                    qx, qy, qz, qw = quaternion_from_euler(3.14, 0.0, -1.57)
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Republished {teamid}_fertiliser_can in base_link")
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for aruco {aruco_info['id']}: {e}")


def main(args=None):
    '''
    Purpose:
        Main function to start both Detection and ArucoTF nodes concurrently.
    '''
    rclpy.init(args=args)

    # Initialize both nodes
    detection_node = Detection()
    aruco_node = ArucoTF()

    # Use MultiThreadedExecutor so callbacks from both nodes can run in parallel
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(detection_node)
    executor.add_node(aruco_node)

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

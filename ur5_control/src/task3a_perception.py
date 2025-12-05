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
        and publishes corresponding transforms in both camera and base_link frames.
    '''
 
    def __init__(self):
        '''
        Purpose:
            Constructor to initialize the ROS2 node, CvBridge, subscriptions, and TF utilities.

        Input Arguments:
            None

        Returns:
            None
        '''
        super().__init__('image_depth_subscriber')

        # Initialize CvBridge for ROS <-> OpenCV conversions
        self.bridge = CvBridge()   # Converts between ROS Image messages and OpenCV images
        self.cv_image = None       # Stores RGB image from camera
        self.depth_image = None    # Stores Depth image from camera

        # Camera intrinsics (defaults until CameraInfo is received)
        self.fx, self.fy = 915.30, 914.03   # Focal lengths
        self.cx, self.cy = 642.72, 361.97   # Principal points

        # TF buffer, listener, broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Topics
        self.rgb_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        self.caminfo_topic = '/camera/camera/color/camera_info'

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.caminfo_topic, self.camera_info_callback, 10)

        self.get_logger().info("Subscribed to RGB, Depth, and Camera Info topics.")

    def camera_info_callback(self, msg: CameraInfo):
        '''
        Purpose:
            Update camera intrinsics from CameraInfo topic.

        Input Arguments:
            msg : CameraInfo
                Contains intrinsic calibration parameters of the camera.

        Returns:
            None
        '''
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg: Image):
        '''
        Purpose:
            Callback for RGB image subscription.
            Converts ROS image to OpenCV format and runs bad fruit detection.

        Input Arguments:
            msg : Image
                ROS2 Image message containing RGB data.

        Returns:
            None
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # cv2.imshow("RGB Image", self.cv_image)

            if self.depth_image is not None:
                self.bad_fruit_detection(self.cv_image, self.depth_image)

            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error in image_callback: {e}")

    def depth_callback(self, msg: Image):
        '''
        Purpose:
            Callback for Depth image subscription.
            Converts ROS depth image to OpenCV format for visualization.

        Input Arguments:
            msg : Image
                ROS2 Image message containing Depth data.

        Returns:
            None
        '''
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype('uint8')

            # cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error in depth_callback: {e}")

    def bad_fruit_detection(self, rgb_image, depth_image):
        '''
        Purpose:
            Detect bad fruits in RGB + Depth images, calculate their 3D position,
            and publish transforms.

        Input Arguments:
            rgb_image : numpy.ndarray
                RGB image from camera.
            depth_image : numpy.ndarray
                Depth image from camera.

        Returns:
            bad_fruits : list
                List of dictionaries containing information of detected bad fruits.

        Example call:
            self.bad_fruit_detection(self.cv_image, self.depth_image)
        '''
        output = rgb_image.copy()

        # ROI for fruit detection
        x1, y1, w1, h1 = 0, 180, 350, 450
        detection_region = output[y1:y1 + h1, x1:x1 + w1]
        hsv = cv2.cvtColor(detection_region, cv2.COLOR_BGR2HSV)

        # Color thresholds
        fruit_lower = np.array([60, 90, 100])
        fruit_upper = np.array([70, 100, 200])
        fruit_top_lower = np.array([35, 40, 80])
        fruit_top_upper = np.array([85, 255, 255])

        grey_mask = cv2.inRange(hsv, fruit_lower, fruit_upper)
        green_mask = cv2.inRange(hsv, fruit_top_lower, fruit_top_upper)

        grey_contours, _ = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        grey_boxes = [cv2.boundingRect(c) for c in grey_contours]

        bad_fruits = []
        fruit_id = 1

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

            # Draw detected fruit
            cv2.rectangle(output, (full_cx - w//2, full_cy - h//2),
                          (full_cx + w//2, full_cy + h//2), (0, 255, 0), 2)
            cv2.circle(output, (full_cx, full_cy), 5, (0, 0, 255), -1)
            cv2.putText(output, f"bad_fruit_{fruit_id}", (full_cx, full_cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            fruit_info = {
                "id": fruit_id,
                "center": (full_cx, full_cy),
                "distance": Z,
                "position_3d": (Z, -X, -Y),
                "width": w
            }

            bad_fruits.append(fruit_info)
            self.publish_tf(fruit_info)
            self.republish_badfruit_in_base(fruit_info['id'])

            fruit_id += 1

        # cv2.imshow("Detected Bad Fruits", output)
        cv2.waitKey(1)
        return bad_fruits

    def is_overlapping(self, box1, box2):
        '''
        Purpose:
            Check if two bounding boxes overlap.

        Input Arguments:
            box1 : tuple
                Bounding box (x, y, w, h).
            box2 : tuple
                Bounding box (x, y, w, h).

        Returns:
            bool : True if boxes overlap, False otherwise.
        '''
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (x1 < x2 + w2 and x1 + w1 > x2) and (y1 < y2 + h2 and y1 + h1 > y2)
    

    def publish_tf(self, fruit_info):
        '''
        Purpose:
            Publish detected fruit transform relative to camera frame.

        Input Arguments:
            fruit_info : dict
                Dictionary containing fruit ID and 3D position.

        Returns:
            None
        '''
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = f"cam{fruit_info['id']}"

        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = fruit_info['position_3d']
        qx, qy, qz, qw = quaternion_from_euler(1.571, 0.0, 1.571)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

        self.tf_broadcaster.sendTransform(t)

    def republish_badfruit_in_base(self, fruit_id, teamid='1425'):
        '''
        Purpose:
            Republish fruit transform relative to base_link frame.

        Input Arguments:
            fruit_id : int
                ID of detected fruit.
            teamid : str
                Team ID prefix for frame naming (default: '1425').

        Returns:
            None
        '''
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
            self.get_logger().info(f"Republished cam{fruit_id} in base_link")
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")



class ArucoTF(Node):

    def __init__(self):
        super().__init__('aruco_tf_publisher')             
        self.color_cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                          # depth raw image variable (from depthimagecb())
        self.bridge = CvBridge()                                                       # OpenCV <-> ROS Image message converter


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
    

    def camera_info_callback(self, msg: CameraInfo):
            '''
            Purpose:
                Update camera intrinsics from CameraInfo topic.

            Input Arguments:
                msg : CameraInfo
                    Contains intrinsic calibration parameters of the camera.

            Returns:
                None
            '''
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]

    def colorimagecb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RGB Image", self.cv_image)

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
        parameters = cv2.aruco.DetectorParameters()

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
                        "yaw": float(angle_aruco)
                    }

                    aruco_info.append(aruco_data)
                    self.aruco_publish_tf(aruco_data)
                    self.republish_aruco_in_base(aruco_data)
            cv2.imshow("Aruco Detection", output)
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
        qx, qy, qz, qw = quaternion_from_euler(1.571, 0.0, 1.571)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (qx, qy, qz, qw)

        self.tf_broadcaster.sendTransform(t)

    def republish_aruco_in_base(self, aruco_info, teamid='1425'):
            try:
                trans = self.tf_buffer.lookup_transform(
                    "base_link",
                    f"camera_3",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5)
                )

                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "base_link"
                t.child_frame_id = f"{teamid}_fertiliser_can_1"
                t.transform.translation = trans.transform.translation

                qx, qy, qz, qw = quaternion_from_euler(1.571, 3.14, 0.0)
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
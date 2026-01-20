#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math

# ==============================================================================
# 1. FEATURE EXTRACTOR (MATH LAYER)
# ==============================================================================
class FeatureDetection:
    def __init__(self):
        self.DETECTION_RADIUS = 0.9  # Increased slightly to see ahead
        self.DELTA = 0.02
        self.EPSILON = 0.05
        self.MAX_SEGMENT_LEN = 1.0
        self.SNUM = 15
        self.P_MIN = 5
        self.SMOOTHING_WIN = 10

    def apply_smoothing(self, x, y, window_size=10):
        if len(x) < window_size: return x, y
        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(x, kernel, mode='same')
        y_smooth = np.convolve(y, kernel, mode='same')
        dist = np.hypot(x - x_smooth, y - y_smooth)
        mask = dist > 0.10
        x_final = np.where(mask, x, x_smooth)
        y_final = np.where(mask, y, y_smooth)
        return x_final, y_final

    def laser_points_set(self, data):
        ranges = np.array(data.ranges)
        angles = np.linspace(data.angle_min, data.angle_max, len(ranges))
        deg_angles = np.degrees(angles)
        
        # Filter for Left and Right sectors (Side-facing)
        angle_mask = ((deg_angles >= -90) & (deg_angles <= -30)) | \
                     ((deg_angles >= 30) & (deg_angles <= 90))
                      
        max_dist = min(data.range_max, self.DETECTION_RADIUS)
        range_mask = (ranges < max_dist) & (ranges > data.range_min)
        valid = angle_mask & range_mask
        
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        
        if len(x) > 5: x, y = self.apply_smoothing(x, y, self.SMOOTHING_WIN)
        return np.vstack((x, y))

    def fast_fit(self, x, y):
        mean_x = np.mean(x); mean_y = np.mean(y)
        data = np.vstack((x - mean_x, y - mean_y))
        try:
            cov = np.cov(data)
            vals, vecs = np.linalg.eigh(cov)
            nx, ny = vecs[:, 0]
            C = -(nx * mean_x + ny * mean_y)
            return nx, ny, C
        except: return 0, 0, 0

    def dist_point_to_line(self, points, line_params):
        nx, ny, C = line_params
        return np.abs(nx * points[0] + ny * points[1] + C)

    def detect_lines(self, points):
        lines = []
        num = points.shape[1]
        if num < self.P_MIN: return []
        i = 0
        while i < num - self.SNUM:
            seed_indices = slice(i, i + self.SNUM)
            seed_pts = points[:, seed_indices]
            params = self.fast_fit(seed_pts[0], seed_pts[1])
            if params[0] == 0: i += 1; continue
            if np.any(self.dist_point_to_line(seed_pts, params) > self.DELTA): i += 1; continue
            j = i + self.SNUM
            line_end_idx = j
            while j < num:
                pt = points[:, j:j+1]
                if self.dist_point_to_line(pt, params) < self.DELTA: line_end_idx = j; j += 1
                else: break
            if (line_end_idx - i) >= self.P_MIN:
                final_pts = points[:, i:line_end_idx]
                p_start = (final_pts[0, 0], final_pts[1, 0])
                p_end = (final_pts[0, -1], final_pts[1, -1])
                length = math.hypot(p_end[0]-p_start[0], p_end[1]-p_start[1])
                if length > self.EPSILON and length < self.MAX_SEGMENT_LEN:
                    lines.append((p_start, p_end))
                    i = line_end_idx
                else: i = line_end_idx
            else: i += 1
        return lines

# ==============================================================================
# 2. SHAPE DETECTOR NODE
# ==============================================================================
class PlantDetectionNode(Node):
    def __init__(self):
        super().__init__('plant_detection_node')

        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        # --- OUTPUT: Publishes shape info for the Nav Node ---
        self.pub_shape_info = self.create_publisher(String, '/shape_info', 10)

        self.det = FeatureDetection()

        # --- MISSION CONFIGURATION ---
        self.TARGET_SEQUENCE = ["PENTAGON", "SQUARE", "TRIANGLE",  "SQUARE", "TRIANGLE"]
        # Cooldowns to prevent re-detecting the same shape immediately
        self.RELEASE_TIMES = [8.0, 42.0, 30.0, 15.0] 
        
        self.INITIAL_DELAY = 5.0
        self.node_start_time = self.get_clock().now()
        self.initial_wait_done = False

        self.current_seq_idx = 0
        self.is_cooling_down = False
        self.cooldown_start_time = None

        # Map Config (Swapped Axes based on your waypoints)
        self.ROW_SPLIT_Y = 0.0
        self.X_BOUNDARIES = [
            (1.0, 1.6), (1.8, 2.4), (2.6, 3.2), (3.3, 3.9)
        ]
        self.LIDAR_OFFSET_X = 0.0
        self.LIDAR_OFFSET_Y = 0.0
        self.robot_pose = None
        self.frame_count = 0

        self.get_logger().info(f"Shape Detector Ready. Targets: {len(self.TARGET_SEQUENCE)}")

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose = (pos.x, pos.y, yaw)

    def scan_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % 3 != 0 or self.robot_pose is None: return

        # 1. Check Initial Delay
        if not self.initial_wait_done:
            elapsed = (self.get_clock().now() - self.node_start_time).nanoseconds * 1e-9
            if elapsed > self.INITIAL_DELAY: self.initial_wait_done = True
            else: return

        # 2. Check Cooldown
        if self.is_cooling_down:
            elapsed = (self.get_clock().now() - self.cooldown_start_time).nanoseconds * 1e-9
            try: required_wait = self.RELEASE_TIMES[self.current_seq_idx - 1]
            except: required_wait = 10.0
            
            if elapsed > required_wait:
                self.is_cooling_down = False
                self.get_logger().info(f"Cooldown done. Searching for: {self.TARGET_SEQUENCE[self.current_seq_idx]}")
            else:
                return 

        # 3. Check Mission Complete
        if self.current_seq_idx >= len(self.TARGET_SEQUENCE): return

        # 4. Feature Extraction
        pc = self.det.laser_points_set(msg)
        if pc.shape[1] < 5: return
        lines = self.det.detect_lines(pc)
        raw_objects = self.classify_objects(lines)

        # 5. Check against Target
        target_shape = self.TARGET_SEQUENCE[self.current_seq_idx]
        
        for name, local_pos, (gx, gy), color, dist in raw_objects:
            if target_shape in name:
                # Identify Label
                plant_id = self.identify_plant(gx, gy)
                label = "UNKNOWN"
                if "SQUARE" in name: label = "BAD_HEALTH"
                elif "TRIANGLE" in name: label = "FERTILIZER_REQUIRED"
                elif "PENTAGON" in name: label = "DOCK_STATION"

                # --- NEW: FORCE ID 0 FOR DOCK STATION ---
                if "DOCK" in label:
                    plant_id = "0"
                # ----------------------------------------

                # Publish Info
                msg_str = f"{label},{gx:.2f},{gy:.2f},{plant_id}"
                out_msg = String()
                out_msg.data = msg_str
                self.pub_shape_info.publish(out_msg)

                self.get_logger().info(f">>> FOUND {target_shape} (ID={plant_id}). Sent to Nav.")

                # Cooldown
                self.current_seq_idx += 1
                self.is_cooling_down = True
                self.cooldown_start_time = self.get_clock().now()
                break

    def identify_plant(self, gx, gy):
        is_top_row = gy < self.ROW_SPLIT_Y
        plant_col = -1
        for idx, (x_min, x_max) in enumerate(self.X_BOUNDARIES):
            if x_min <= gx <= x_max:
                plant_col = idx
                break
        if plant_col == -1: return "0"
        return str(plant_col + 1) if is_top_row else str(plant_col + 5)

    def get_global_coords(self, local_x, local_y):
        rx, ry, theta = self.robot_pose
        base_x = local_x + self.LIDAR_OFFSET_X
        base_y = local_y + self.LIDAR_OFFSET_Y
        gx = rx + (base_x * math.cos(theta) - base_y * math.sin(theta))
        gy = ry + (base_x * math.sin(theta) + base_y * math.cos(theta))
        return gx, gy

    def get_line_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))

    def get_angle(self, v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(det, dot)
        deg = abs(math.degrees(angle))
        if deg > 180: deg = 360 - deg
        return deg

    def classify_objects(self, lines):
        detected_objects = []
        potential_corners = []

        # Find intersections (Corners)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                l1_start, l1_end = lines[i]
                l2_start, l2_end = lines[j]

                intersection = self.get_line_intersection(l1_start, l1_end, l2_start, l2_end)
                if intersection is None: continue
                cx, cy = intersection

                # Check if intersection is valid (near segments)
                dist_l1 = min(math.hypot(cx-l1_start[0], cy-l1_start[1]), math.hypot(cx-l1_end[0], cy-l1_end[1]))
                dist_l2 = min(math.hypot(cx-l2_start[0], cy-l2_start[1]), math.hypot(cx-l2_end[0], cy-l2_end[1]))

                if dist_l1 < 0.3 and dist_l2 < 0.3:
                    v1 = (l1_end[0] - l1_start[0], l1_end[1] - l1_start[1])
                    v2 = (l2_end[0] - l2_start[0], l2_end[1] - l2_start[1])
                    angle = self.get_angle(v1, v2)
                    if angle > 90: angle = 180 - angle

                    name = "Unknown"
                    if 75 <= angle <= 105: name = "SQUARE"
                    elif 15 <= angle <= 50: name = "TRIANGLE"

                    if name != "Unknown":
                        potential_corners.append({'name': name, 'pos': (cx, cy), 'lines': {i, j}})

        # Merge corners
        processed_indices = set()
        for m in range(len(potential_corners)):
            for n in range(m + 1, len(potential_corners)):
                c1 = potential_corners[m]
                c2 = potential_corners[n]

                if not c1['lines'].isdisjoint(c2['lines']):
                    mid_x = (c1['pos'][0] + c2['pos'][0]) / 2
                    mid_y = (c1['pos'][1] + c2['pos'][1]) / 2
                    gx, gy = self.get_global_coords(mid_x, mid_y)
                    dist = math.hypot(mid_x, mid_y)
                    obj = None
                    # "House" Logic (Square + Triangle = Pentagon)
                    if (c1['name'] == "SQUARE" and c2['name'] == "TRIANGLE") or \
                       (c1['name'] == "TRIANGLE" and c2['name'] == "SQUARE"):
                        obj = ("PENTAGON (Verified)", (mid_x, mid_y), (gx, gy), (0, 0, 0), dist)
                    # Double Square Logic
                    elif c1['name'] == "SQUARE" and c2['name'] == "SQUARE":
                         obj = ("SQUARE (Verified)", (mid_x, mid_y), (gx, gy), (0, 0, 0), dist)

                    if obj:
                        detected_objects.append(obj)
                        processed_indices.add(m); processed_indices.add(n)

        # Add single corners
        for idx, c in enumerate(potential_corners):
            if idx not in processed_indices:
                if c['name'] == "TRIANGLE":
                    gx, gy = self.get_global_coords(c['pos'][0], c['pos'][1])
                    dist = math.hypot(c['pos'][0], c['pos'][1])
                    detected_objects.append((c['name'], c['pos'], (gx, gy), (0, 0, 0), dist))
                # Add Single Squares if robust
                if c['name'] == "SQUARE":
                     gx, gy = self.get_global_coords(c['pos'][0], c['pos'][1])
                     dist = math.hypot(c['pos'][0], c['pos'][1])
                     detected_objects.append((c['name'], c['pos'], (gx, gy), (0, 0, 0), dist))

        return detected_objects

def main():
    rclpy.init()
    try: rclpy.spin(PlantDetectionNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()

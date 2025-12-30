#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math

# ==============================================================================
# 1. OPTIMIZED FEATURE EXTRACTOR
# ==============================================================================
class FeatureDetection:
    def __init__(self):
        self.DETECTION_RADIUS = 0.9
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
        
        # Filter for Left and Right sectors (Side-facing logic)
        angle_mask = ((deg_angles >= -90) & (deg_angles <= -45)) | \
                     ((deg_angles >= 45) & (deg_angles <= 90))
                     
        max_dist = min(data.range_max, self.DETECTION_RADIUS)
        range_mask = (ranges < max_dist) & (ranges > data.range_min)
        valid = angle_mask & range_mask
        
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        
        if len(x) > 5: x, y = self.apply_smoothing(x, y, self.SMOOTHING_WIN)
        return np.vstack((x, y))

    def dist_point_to_line(self, points, line_params):
        nx, ny, C = line_params
        return np.abs(nx * points[0] + ny * points[1] + C)

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
# 2. PLANT & SHAPE DETECTOR NODE (HARDWARE)
# ==============================================================================
class PlantDetectionNode(Node):
    def __init__(self):
        super().__init__('plant_detection_node')

        # REMOVED: use_sim_time parameter
        # REMOVED: Custom QoS Profile (Hardware drivers usually publish Reliable)

        # Using standard depth=10 (Reliable) for subscriptions
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.pub_status = self.create_publisher(String, '/detection_status', 10)

        self.det = FeatureDetection()

        # --- CONFIGURATION ---
        self.TARGET_SEQUENCE = ["PENTAGON", "SQUARE", "TRIANGLE",  "SQUARE", "TRIANGLE",]
        self.RELEASE_TIMES = [15.0, 10.0, 10.0, 15.0]

        self.current_seq_idx = 0
        self.target_locked = False
        self.published_count = 0
        self.lock_time = None
        self.stored_shape_data = None

        self.ALIGNMENT_TOLERANCE = 0.08

        # Map Config (Swapped Axes)
        self.ROW_SPLIT_Y = 0.0
        self.X_BOUNDARIES = [
            (1.0, 1.6), (1.8, 2.4), (2.6, 3.2), (3.3, 3.9)
        ]
        self.LIDAR_OFFSET_X = 0.4
        self.LIDAR_OFFSET_Y = 0.0
        self.robot_pose = None
        self.frame_count = 0

        self.get_logger().info(f"HARDWARE Detector Ready. Sequence Length: {len(self.TARGET_SEQUENCE)}")

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose = (pos.x, pos.y, yaw)

    def scan_cb(self, msg):
        self.frame_count += 1
        # Throttle processing (process every 2nd frame)
        if self.frame_count % 2 != 0 or self.robot_pose is None: return

        pc = self.det.laser_points_set(msg)
        lines = []
        if pc.shape[1] > 0:
            lines = self.det.detect_lines(pc)

        raw_objects = self.classify_objects(lines)
        self.manage_sequence_and_publish(raw_objects)

    def identify_plant(self, gx, gy):
        is_top_row = gy > self.ROW_SPLIT_Y
        plant_col = -1
        for idx, (x_min, x_max) in enumerate(self.X_BOUNDARIES):
            if x_min <= gx <= x_max:
                plant_col = idx
                break
        if plant_col == -1: return "0"
        return str(plant_col + 1) if is_top_row else str(plant_col + 5)

    def manage_sequence_and_publish(self, detected_objects):
        if self.current_seq_idx >= len(self.TARGET_SEQUENCE): return

        target_shape = self.TARGET_SEQUENCE[self.current_seq_idx]
        try: current_release_time = self.RELEASE_TIMES[self.current_seq_idx]
        except IndexError: current_release_time = 15.0

        # --- STEP 1: LATCH ONTO FIRST SIGHTING ---
        if self.stored_shape_data is None:
            for name, local_pos, (gx, gy), color, dist in detected_objects:
                if target_shape in name:
                    plant_id = self.identify_plant(gx, gy)
                    label = "UNKNOWN"
                    if "SQUARE" in name: label = "BAD_HEALTH"
                    elif "TRIANGLE" in name: label = "FERTILIZER_REQUIRED"
                    elif "PENTAGON" in name: label = "DOCK_STATION"

                    self.stored_shape_data = {
                        'gx': gx, 'gy': gy, 'plant_id': plant_id, 'label': label
                    }
                    self.get_logger().info(f"VISUAL CONTACT: {target_shape} at X={gx:.2f}. Waiting for alignment...")
                    break

        # --- STEP 2: CHECK ALIGNMENT ---
        if self.stored_shape_data is not None and not self.target_locked:
            rx, ry, _ = self.robot_pose
            shape_x = self.stored_shape_data['gx']
            x_diff = abs(rx - shape_x)

            if x_diff < self.ALIGNMENT_TOLERANCE:
                if self.published_count < 1:
                    label = self.stored_shape_data['label']
                    plant_id = self.stored_shape_data['plant_id']

                    msg = String()
                    data_str = f"{label},{shape_x:.2f},{ry:.2f},{plant_id}"
                    msg.data = data_str
                    self.pub_status.publish(msg)

                    self.get_logger().info(f"ALIGNED -> PUBLISHED: {data_str}")
                    self.published_count += 1

                    self.target_locked = True
                    self.lock_time = self.get_clock().now()

        # --- STEP 3: COOLDOWN & NEXT TASK ---
        if self.target_locked:
            elapsed = (self.get_clock().now() - self.lock_time).nanoseconds * 1e-9
            if elapsed > current_release_time:
                self.get_logger().info(f"TASK COMPLETE ({target_shape}). Scanning for next target...")
                self.current_seq_idx += 1
                self.target_locked = False
                self.lock_time = None
                self.published_count = 0
                self.stored_shape_data = None

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

                # Check if intersection is actually near the segments
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

        # Merge corners to find whole shapes
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
                    if c1['name'] == "SQUARE" and c2['name'] == "SQUARE":
                        obj = ("SQUARE (Verified)", (mid_x, mid_y), (gx, gy), (0, 0, 0), dist)
                    elif c1['name'] == "TRIANGLE" and c2['name'] == "TRIANGLE":
                        obj = ("PENTAGON (Verified)", (mid_x, mid_y), (gx, gy), (0, 0, 0), dist)
                    elif (c1['name'] == "SQUARE" and c2['name'] == "TRIANGLE") or \
                         (c1['name'] == "TRIANGLE" and c2['name'] == "SQUARE"):
                        obj = ("PENTAGON (Verified)", (mid_x, mid_y), (gx, gy), (0, 0, 0), dist)

                    if obj:
                        detected_objects.append(obj)
                        processed_indices.add(m); processed_indices.add(n)

        # Add remaining single corners as potential objects
        for idx, c in enumerate(potential_corners):
            if idx not in processed_indices:
                if c['name'] == "TRIANGLE":
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

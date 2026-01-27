#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math

# ==================== FEATURE EXTRACTOR (Standard) ====================
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
        angle_mask = ((deg_angles >= -90) & (deg_angles <= -45)) | \
                     ((deg_angles >= 45) & (deg_angles <= 90))
        max_dist = min(data.range_max, self.DETECTION_RADIUS)
        range_mask = (ranges < max_dist) & (ranges > data.range_min)
        valid = angle_mask & range_mask
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        if len(x) > 5: x, y = self.apply_smoothing(x, y, self.SMOOTHING_WIN)
        return np.vstack((x, y))

    def detect_lines(self, points):
        # Simplified for brevity, same robust logic as before
        lines = []
        num = points.shape[1]
        if num < self.P_MIN: return []
        
        def dist_point_to_line(pts, p): 
            return np.abs(p[0]*pts[0] + p[1]*pts[1] + p[2])
        def fast_fit(x, y):
            mx, my = np.mean(x), np.mean(y)
            data = np.vstack((x-mx, y-my))
            try:
                vals, vecs = np.linalg.eigh(np.cov(data))
                nx, ny = vecs[:, 0]
                return nx, ny, -(nx*mx + ny*my)
            except: return 0,0,0

        i = 0
        while i < num - self.SNUM:
            seed = points[:, i:i+self.SNUM]
            params = fast_fit(seed[0], seed[1])
            if params[0]==0 or np.any(dist_point_to_line(seed, params)>self.DELTA):
                i+=1; continue
            
            j = i + self.SNUM
            line_end = j
            while j < num:
                if dist_point_to_line(points[:, j:j+1], params) < self.DELTA:
                    line_end = j; j+=1
                else: break
            
            if (line_end - i) >= self.P_MIN:
                final = points[:, i:line_end]
                p1, p2 = (final[0,0], final[1,0]), (final[0,-1], final[1,-1])
                length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                if self.EPSILON < length < self.MAX_SEGMENT_LEN:
                    lines.append((p1, p2))
                    i = line_end
                else: i = line_end
            else: i += 1
        return lines

# ==================== PERCEPTION NODE (Hardware) ====================
class PlantDetectionNode(Node):
    def __init__(self):
        super().__init__('plant_detection_hw')
        # HARDWARE MODE
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.pub_status = self.create_publisher(String, '/raw_detection', 10)
        
        self.det = FeatureDetection()
        
        # SEQUENCE & TIMING
        self.TARGET_SEQUENCE = ["PENTAGON", "SQUARE", "TRIANGLE", "TRIANGLE", "SQUARE", "TRIANGLE", "PENTAGON"] 
        self.RELEASE_TIMES = [1.0, 42.0, 5.0, 5.0, 3.0, 20.0] 
        self.current_seq_idx = 0
        
        # STATE
        self.target_locked = False 
        self.published_count = 0
        self.lock_time = None
        self.stored_shape_data = None    
        
        # MAPPING (X-Axis Logic)
        self.ROW_SPLIT_X = 0.0 # Y coordinate split
        self.Y_BOUNDARIES = [(1.0, 1.6), (1.8, 2.4), (2.6, 3.2), (3.3, 3.9)] # X coordinate bounds
        
        self.LIDAR_OFFSET_X = 0.4
        self.LIDAR_OFFSET_Y = 0.0 
        self.robot_pose = None
        self.frame_count = 0 
        
        print(f"\n{'='*40}\n[EYE] VISION READY. Looking for: {self.TARGET_SEQUENCE[0]}\n{'='*40}")

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.robot_pose = (p.x, p.y, yaw)

    def scan_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % 2 != 0 or self.robot_pose is None: return

        pc = self.det.laser_points_set(msg)
        lines = []
        if pc.shape[1] > 0: lines = self.det.detect_lines(pc)
        
        objects = self.classify_objects(lines)
        self.manage_logic(objects)

    def identify_plant(self, gx, gy):
        # Hardware: X is forward. "Y boundaries" check X. "Row Split" checks Y.
        is_top_row = gy > self.ROW_SPLIT_X 
        plant_col = -1
        for idx, (min_v, max_v) in enumerate(self.Y_BOUNDARIES):
            if min_v <= gx <= max_v:
                plant_col = idx; break
        if plant_col == -1: return "0"
        return str(plant_col + 1) if is_top_row else str(plant_col + 5)

    def manage_logic(self, objects):
        if self.current_seq_idx >= len(self.TARGET_SEQUENCE): return
        target = self.TARGET_SEQUENCE[self.current_seq_idx]
        
        try: wait_time = self.RELEASE_TIMES[self.current_seq_idx]
        except: wait_time = 15.0

        # --- 1. SEARCH ---
        if self.stored_shape_data is None:
            for name, _, (gx, gy), _, dist in objects:
                # SAFETY FILTER (Critical for Hardware)
                if dist > 1.2: continue 
                
                if target in name:
                    pid = self.identify_plant(gx, gy)
                    lbl = "UNKNOWN"
                    if "SQUARE" in name: lbl = "BAD_HEALTH"
                    elif "TRIANGLE" in name: lbl = "FERTILIZER_REQUIRED"
                    elif "PENTAGON" in name: lbl = "DOCK_STATION"

                    self.stored_shape_data = {'gx': gx, 'gy': gy, 'pid': pid, 'lbl': lbl}
                    self.target_locked = True
                    self.lock_time = self.get_clock().now()
                    self.published_count = 0
                    
                    self.get_logger().info(f"[EYE] 🔒 LOCK: {name} (X={gx:.2f}, ID={pid})")
                    break

        # --- 2. BURST PUBLISH (Reliability) ---
        if self.target_locked:
            elapsed = (self.get_clock().now() - self.lock_time).nanoseconds * 1e-9
            
            # Spam for 0.5s to ensure Nav hears it
            if elapsed < 0.5:
                d = self.stored_shape_data
                msg = String()
                # Sending X (gx) as the target for stopping
                msg.data = f"{d['lbl']},{d['gy']:.2f},{d['gx']:.2f},{d['pid']}"
                self.pub_status.publish(msg)
                
                if self.published_count == 0:
                    self.get_logger().info("[EYE] 📨 SENDING STOP REQ (Burst Mode)")
                self.published_count += 1

            # --- 3. COOLDOWN ---
            if elapsed > wait_time:
                self.get_logger().info(f"[EYE] ⏳ WAIT DONE ({wait_time}s). Next: {self.TARGET_SEQUENCE[min(self.current_seq_idx+1, len(self.TARGET_SEQUENCE)-1)]}")
                self.current_seq_idx += 1
                self.target_locked = False
                self.lock_time = None
                self.stored_shape_data = None

    def get_global_coords(self, lx, ly):
        rx, ry, th = self.robot_pose
        bx = lx + self.LIDAR_OFFSET_X
        by = ly + self.LIDAR_OFFSET_Y
        return (rx + bx*math.cos(th) - by*math.sin(th), ry + bx*math.sin(th) + by*math.cos(th))

    def classify_objects(self, lines):
        # ... (Geometry Math - Simplified wrapper for readability) ...
        # Standard intersection & angle logic
        objs = []
        corners = []
        
        def get_intersect(p1, p2, p3, p4):
            x1,y1,x2,y2,x3,y3,x4,y4 = *p1, *p2, *p3, *p4
            d = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
            if d==0: return None
            ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3))/d
            return (x1+ua*(x2-x1), y1+ua*(y2-y1))

        def get_angle(v1, v2):
            ang = math.degrees(math.atan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1]))
            return abs(ang) if abs(ang)<=180 else 360-abs(ang)

        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                inter = get_intersect(*lines[i], *lines[j])
                if not inter: continue
                
                cx, cy = inter
                # Check closeness to endpoints
                d1 = min(math.hypot(cx-lines[i][0][0], cy-lines[i][0][1]), math.hypot(cx-lines[i][1][0], cy-lines[i][1][1]))
                d2 = min(math.hypot(cx-lines[j][0][0], cy-lines[j][0][1]), math.hypot(cx-lines[j][1][0], cy-lines[j][1][1]))
                
                if d1<0.3 and d2<0.3:
                    v1 = (lines[i][1][0]-lines[i][0][0], lines[i][1][1]-lines[i][0][1])
                    v2 = (lines[j][1][0]-lines[j][0][0], lines[j][1][1]-lines[j][0][1])
                    ang = get_angle(v1, v2)
                    if ang>90: ang = 180-ang
                    
                    name = "SQUARE" if 75<=ang<=105 else "TRIANGLE" if 15<=ang<=50 else None
                    if name: corners.append({'name':name, 'pos':(cx,cy), 'ids':{i,j}})

        # Merge logic (Square + Triangle = Pentagon)
        used = set()
        for m in range(len(corners)):
            for n in range(m+1, len(corners)):
                c1, c2 = corners[m], corners[n]
                if not c1['ids'].isdisjoint(c2['ids']):
                    mx, my = (c1['pos'][0]+c2['pos'][0])/2, (c1['pos'][1]+c2['pos'][1])/2
                    gx, gy = self.get_global_coords(mx, my)
                    dist = math.hypot(mx, my)
                    
                    pair = tuple(sorted([c1['name'], c2['name']]))
                    final_name = "PENTAGON" if pair==("SQUARE","TRIANGLE") or pair==("TRIANGLE","TRIANGLE") else c1['name']
                    
                    objs.append((final_name, (mx,my), (gx,gy), None, dist))
                    used.update([m,n])
        
        # Add remaining single corners
        for idx, c in enumerate(corners):
            if idx not in used and c['name'] == "TRIANGLE":
                gx, gy = self.get_global_coords(*c['pos'])
                dist = math.hypot(*c['pos'])
                objs.append((c['name'], c['pos'], (gx,gy), None, dist))

        return objs

def main():
    rclpy.init()
    try: rclpy.spin(PlantDetectionNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()

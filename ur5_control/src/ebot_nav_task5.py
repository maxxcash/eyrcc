#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

# -------------------------- WAYPOINTS ---------------------------
WAYPOINTS = [
    (0.0, 0.0, -1.3),
    (0.2, -1.67, 0.0),
    (5.0, -1.65, 1.571),
    (4.8,  1.78, 3.14),
    (0.1,  1.78, -1.57),
    (0.1, 0.0, -3.14),
    (5.0, 0.0, 0.0)
]

FINAL_REVERSE_X_TARGET = 0.2

# ------------------------ TUNING PARAMETERS -----------------------
WAIT_BEFORE_PUBLISH = 1.0 
# WAIT_AFTER_PUBLISH is now dynamic!

POSE_TOL = 0.05
YAW_TOL = math.radians(15)
START_DRIVE_ANGLE = math.radians(15)
HOLD_TIME = 0.4
LOOP_HZ = 30.0

# PID / Motion Constraints
KP_LIN = 1.1; KP_ANG = 1.5
MAX_LIN = 0.5; MAX_ANG = 1.0
MAX_LIN_ACCEL = 0.8; MAX_ANG_ACCEL = 3.0
MIN_LIN = 0.08; MIN_ANG = 0.15
ALPHA = 0.55

# -------------------------- HELPERS ---------------------------
def q_to_yaw(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

def normalize(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def clamp(x, a, b):
    return max(a, min(b, x))

class WaypointNav(Node):

    def __init__(self):
        super().__init__("ebot_nav_logger")

        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.sub_detection = self.create_subscription(String, "/shape_info", self.cb_shape_info, 10)
        
        self.pub_status = self.create_publisher(String, "/detection_status", 10)
        self.pub_cmd  = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(1.0 / LOOP_HZ, self.loop)

        # Robot State
        self.x = None; self.y = None; self.yaw = None
        self.pose_ready = False

        # Navigation State
        self.wi = 0
        self.state = "INIT"  
        self.hold_start = None
        self.prev_lin_cmd = 0.0
        self.prev_ang_cmd = 0.0
        self.scan = []

        # --- PAUSE LOGIC VARIABLES ---
        self.target_x = None         
        self.pending_msg_data = ""   
        self.stop_condition = 0      
        
        self.pause_phase = 0         
        self.phase_start_time = 0.0
        
        # --- NEW: Dynamic Wait Time ---
        self.dynamic_wait_time = 2.0 

        self.last_time = self.time_now()
        self.get_logger().info(">> NAV NODE READY. Waiting for Odom...")

    def time_now(self):
        t = self.get_clock().now()
        s, ns = t.seconds_nanoseconds()
        return s + ns*1e-9

    # ----------------------------- DETECTION CALLBACK -----------------------------
    def cb_shape_info(self, msg):
        data = msg.data
        if self.pause_phase > 0 or self.target_x is not None: return

        try:
            parts = data.split(',')
            if len(parts) >= 3:
                label = parts[0].strip()
                detected_x = float(parts[1]) 

                valid_triggers = ["DOCK", "FERT", "BAD", "bad", "dock", "fert"]

                if any(x in label for x in valid_triggers):
                    self.target_x = detected_x
                    self.pending_msg_data = data 

                    # --- NEW: SET DURATION BASED ON TYPE ---
                    if "DOCK" in label:
                        # 1s (Phase 1) + 9s (Phase 2) = 10s Total
                        self.dynamic_wait_time = 15.0
                        self.get_logger().info("--> DOCK DETECTED: Setting Long Wait (10s)")
                    else:
                        # 1s (Phase 1) + 2s (Phase 2) = 3s Total
                        self.dynamic_wait_time = 2.0
                    # ---------------------------------------

                    if self.target_x > self.x:
                        self.stop_condition = 1 
                    else:
                        self.stop_condition = -1 

                    self.get_logger().info(f"+++ SHAPE SEEN: '{label}' at X={self.target_x:.2f} +++")

        except ValueError: pass

    # ----------------------------- ODOM & SCAN -----------------------------
    def cb_odom(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pyaw = q_to_yaw(msg.pose.pose.orientation)
        
        if not self.pose_ready:
            self.x, self.y, self.yaw = px, py, pyaw
            self.pose_ready = True
            self.get_logger().info(f">> ODOM RECEIVED. Starting at X={self.x:.2f}")
            self.state = "ROTATE" 
            self.log_waypoint_start()
        else:
            self.x = ALPHA*self.x + (1-ALPHA)*px
            self.y = ALPHA*self.y + (1-ALPHA)*py
            self.yaw = normalize(self.yaw + (1-ALPHA)*normalize(pyaw - self.yaw))

    def cb_scan(self, m): self.scan = list(m.ranges)

    # ----------------------------- MAIN LOOP -----------------------------
    def loop(self):
        if not self.pose_ready: return

        now = self.time_now()
        dt = now - self.last_time
        if dt <= 0: dt = 1.0/LOOP_HZ
        self.last_time = now

        # ================= PAUSE SEQUENCE LOGIC =================
        
        # --- PHASE 1: Wait 1 sec BEFORE publishing ---
        if self.pause_phase == 1:
            self.stop()
            if (now - self.phase_start_time) >= WAIT_BEFORE_PUBLISH:
                out_msg = String()
                out_msg.data = self.pending_msg_data
                self.pub_status.publish(out_msg)
                self.get_logger().info(f">> PHASE 1 DONE. PUBLISHED: {self.pending_msg_data}")
                
                self.pause_phase = 2
                self.phase_start_time = now 
                self.get_logger().info(f">> Starting PHASE 2 (Wait {self.dynamic_wait_time}s)...")
            return

        # --- PHASE 2: Wait VARIABLE secs AFTER publishing ---
        if self.pause_phase == 2:
            self.stop()
            # USE DYNAMIC TIME HERE
            if (now - self.phase_start_time) >= self.dynamic_wait_time:
                self.get_logger().info(">> PHASE 2 DONE. Resuming Mission.")
                self.pause_phase = 0
                self.target_x = None
                self.pending_msg_data = ""
            return

        # --- TRIGGER ---
        if self.target_x is not None:
            stop = False
            if self.stop_condition == 1 and self.x >= self.target_x: stop = True
            elif self.stop_condition == -1 and self.x <= self.target_x: stop = True

            if stop:
                self.get_logger().warn(f"!!! TARGET REACHED. STARTING STOP SEQUENCE !!!")
                self.stop()
                self.pause_phase = 1         
                self.phase_start_time = now  
                return
        
        # ================= STANDARD NAVIGATION =================
        
        if self.wi >= len(WAYPOINTS):
            if self.state != "FINAL_REVERSE":
                self.state = "FINAL_REVERSE"
                self.get_logger().info(">> ALL WAYPOINTS DONE. Starting Final Reverse.")

            if self.x > FINAL_REVERSE_X_TARGET:
                self.cmd(self.ramp_linear(-0.50, dt), 0.0)
                return
            else:
                self.stop()
                self.get_logger().info(">> MISSION COMPLETE. Shutting Down.")
                rclpy.shutdown()
                return

        gx, gy, gyaw = WAYPOINTS[self.wi]
        dx = gx - self.x; dy = gy - self.y
        dist = math.hypot(dx, dy)
        yaw_err = normalize(math.atan2(dy, dx) - self.yaw)
        final_yaw_err = normalize(gyaw - self.yaw)

        if self.state == "ROTATE":
            if abs(yaw_err) > START_DRIVE_ANGLE:
                ang = clamp(KP_ANG * yaw_err, -MAX_ANG, MAX_ANG)
                if abs(ang) < MIN_ANG: ang = math.copysign(MIN_ANG, ang)
                self.cmd(0, self.ramp_angular(ang, dt))
            else:
                self.stop()
                self.change_state("DRIVE")

        elif self.state == "DRIVE":
            if dist > POSE_TOL:
                lin = clamp(KP_LIN * dist, MIN_LIN, MAX_LIN)
                ang = clamp(lin * (2*math.sin(yaw_err))/0.6, -MAX_ANG, MAX_ANG)
                self.cmd(self.ramp_linear(lin, dt), self.ramp_angular(ang, dt))
                self.last_yaw_err = final_yaw_err
            else:
                self.stop()
                self.change_state("ALIGN")

        elif self.state == "ALIGN":
            if abs(final_yaw_err) > YAW_TOL:
                rate = (final_yaw_err - self.last_yaw_err) / dt
                ang = clamp(KP_ANG * final_yaw_err + 0.1 * rate, -MAX_ANG, MAX_ANG)
                if abs(ang) < MIN_ANG: ang = math.copysign(MIN_ANG, ang)
                self.cmd(0, ang)
                self.last_yaw_err = final_yaw_err
            else:
                self.stop()
                self.hold_start = now
                self.change_state("HOLD")

        elif self.state == "HOLD":
            self.stop()
            if now - self.hold_start >= HOLD_TIME:
                self.wi += 1
                if self.wi < len(WAYPOINTS):
                    self.log_waypoint_start()
                    self.change_state("ROTATE")

    def change_state(self, new_state):
        self.get_logger().info(f"    State Change: {self.state} -> {new_state}")
        self.state = new_state

    def log_waypoint_start(self):
        gx, gy, _ = WAYPOINTS[self.wi]
        self.get_logger().info(f"--- WAYPOINT [{self.wi+1}/{len(WAYPOINTS)}]: {gx:.2f}, {gy:.2f} ---")

    def cmd(self, l, a):
        m = Twist(); m.linear.x, m.angular.z = float(l), float(a)
        self.pub_cmd.publish(m)
    def stop(self):
        self.pub_cmd.publish(Twist()); self.prev_lin_cmd=0.0; self.prev_ang_cmd=0.0
    def ramp_linear(self, d, dt):
        self.prev_lin_cmd += clamp(d - self.prev_lin_cmd, -MAX_LIN_ACCEL*dt, MAX_LIN_ACCEL*dt)
        return float(self.prev_lin_cmd)
    def ramp_angular(self, d, dt):
        self.prev_ang_cmd += clamp(d - self.prev_ang_cmd, -MAX_ANG_ACCEL*dt, MAX_ANG_ACCEL*dt)
        return float(self.prev_ang_cmd)

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(WaypointNav())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == "__main__":
    main()

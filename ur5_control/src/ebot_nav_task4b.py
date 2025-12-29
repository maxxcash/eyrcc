#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math

# -------------------------- WAYPOINTS ---------------------------
WAYPOINTS = [      
    (0.0, 0.0, 1.17),
    (0.2, -1.65, 0.0), 
    (4.5, -1.65, -1.571),          
    (3.8,  1.6, -3.14),      
    (0.1,  1.6, 1.57),    
    (0.1, 0.0, 0.0),  
    (4.6, 0.0, 0.0)]

FINAL_REVERSE_X_TARGET = 0.0  

# ------------------------ TUNING PARAMETERS -----------------------
STOP_DURATION = 2.0         
POSE_TOL = 0.05
YAW_TOL = math.radians(15)  
START_DRIVE_ANGLE = math.radians(15) 
HOLD_TIME = 0.4
LOOP_HZ = 30.0
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
        
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.sub_detection = self.create_subscription(String, "/detection_status", self.cb_detection, 10)
        self.pub_cmd  = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.timer = self.create_timer(1.0 / LOOP_HZ, self.loop)

        # Robot State
        self.x = None; self.y = None; self.yaw = None
        self.pose_ready = False
        
        # Navigation State
        self.wi = 0
        self.state = "INIT"  # Starting state
        self.hold_start = None
        self.prev_lin_cmd = 0.0
        self.prev_ang_cmd = 0.0
        self.last_yaw_err = 0.0
        self.scan = []
        
        # Alignment State
        self.target_x = None       
        self.target_label = ""     
        self.stop_condition = 0    
        
        self.is_paused = False           
        self.pause_start_time = 0.0

        self.last_time = self.time_now()
        self.get_logger().info(">> SYSTEM READY. Waiting for Odom...")

    def time_now(self):
        t = self.get_clock().now()
        s, ns = t.seconds_nanoseconds()
        return s + ns*1e-9

    # ----------------------------- DETECTION -----------------------------
    def cb_detection(self, msg):
        data = msg.data
        if self.is_paused or self.target_x is not None: return

        try:
            # Expected: "Label, x, y, ID"
            parts = data.split(',')
            if len(parts) >= 3:
                label = parts[0].strip()
                detected_x = float(parts[1])  # Using X for alignment
                
                valid_triggers = ["DOCK", "FERT", "BAD", "bad", "dock", "fert"]
                
                if any(x in label for x in valid_triggers):
                    self.target_x = detected_x
                    self.target_label = label
                    
                    if self.target_x > self.x:
                        self.stop_condition = 1 
                        mode = "FORWARD (Stop >= Target)"
                    else:
                        self.stop_condition = -1 
                        mode = "BACKWARD (Stop <= Target)"
                    
                    self.get_logger().info(f"+++ TARGET LOCKED: '{label}' +++")
                    self.get_logger().info(f"    Target X: {self.target_x:.3f} | Current X: {self.x:.3f}")
                    self.get_logger().info(f"    Logic: {mode}")

        except ValueError: pass

    # ----------------------------- ODOM & SCAN -----------------------------
    def cb_odom(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pyaw = q_to_yaw(msg.pose.pose.orientation)
        if not self.pose_ready:
            self.x, self.y, self.yaw = px, py, pyaw
            self.pose_ready = True
            self.get_logger().info(f">> ODOM RECEIVED. Starting at X={self.x:.2f}, Y={self.y:.2f}")
            self.state = "ROTATE" # Start immediately
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

        # --- 1. PAUSE LOGIC ---
        if self.is_paused:
            self.stop()
            if (now - self.pause_start_time) >= STOP_DURATION:
                self.get_logger().info(f">> RESUMING navigation. Done with '{self.target_label}'.")
                self.is_paused = False
                self.target_x = None  
            return 

        # --- 2. TARGET STOP CHECK ---
        if self.target_x is not None:
            stop = False
            if self.stop_condition == 1 and self.x >= self.target_x: stop = True
            elif self.stop_condition == -1 and self.x <= self.target_x: stop = True

            if stop:
                self.get_logger().warn(f"!!! STOP TRIGGERED !!!")
                self.get_logger().warn(f"    Robot X: {self.x:.4f} crossed Target: {self.target_x:.4f}")
                self.get_logger().warn(f"    Action: Pausing for {STOP_DURATION}s")
                self.stop()
                self.is_paused = True
                self.pause_start_time = now
                return

        # --- 3. WAYPOINT NAV ---
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

        # STATE MACHINE LOGGING
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

    # --- HELPERS ---
    def change_state(self, new_state):
        self.get_logger().info(f"    State Change: {self.state} -> {new_state}")
        self.state = new_state

    def log_waypoint_start(self):
        gx, gy, _ = WAYPOINTS[self.wi]
        self.get_logger().info(f"--------------------------------------------------")
        self.get_logger().info(f"WAYPOINT [{self.wi+1}/{len(WAYPOINTS)}]: Moving to X={gx:.2f}, Y={gy:.2f}")
        self.get_logger().info(f"--------------------------------------------------")

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

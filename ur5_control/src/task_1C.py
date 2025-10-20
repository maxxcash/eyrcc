#!/usr/bin/env python3
'''
# Team ID:         4686
# Theme:           Krishi coBot
# Author List:     Sahil Ranveer
# Filename:        task_1C.py
# Functions:       FinalArmController.__init__, FinalArmController.joint_states_callback, 
#                  FinalArmController.calculate_all_ik_solutions, FinalArmController.control_loop, main
# Global variables: JOINT_ORDER
'''

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.srv import GetPositionIK
import numpy as np
import time
import math

# JOINT_ORDER: Defines the expected order of joints for calculations and commands.
JOINT_ORDER = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]

class FinalArmController(Node):
    '''
    Purpose:
    ---
    Controls a UR5e robot arm. It first moves to a known-good joint staging pose to ensure
    reliable operation, then uses an Inverse Kinematics (IK) solver to calculate and move
    to a sequence of predefined Cartesian waypoints for the competition task.
    '''
    def __init__(self):
        '''
        Initializes the ROS 2 node, publishers, subscribers, and state machine.
        '''
        super().__init__('final_arm_controller')

        # --- ROS 2 Communications ---
        self.publisher = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        
        # --- State Machine & Robot State ---
        self.state = 'INITIALIZING'
        self.current_joint_angles = None
        self.joint_names_in_message = None
        self.active_target_angles = None

        # --- Staging Pose (in Joint Space) ---
        # This pose is easily reachable and prepares the arm for reliable IK calculations.
        self.staging_joint_angles = np.array([0.0, -2.0, 2.0, -1.57, -1.57, 0.0])

        # --- Cartesian Waypoint Definitions (from task description) ---
        self.cartesian_waypoints = [
            Pose(position=Point(x=-0.214, y=-0.532, z=0.557), orientation=Quaternion(x=0.707, y=0.028, z=0.034, w=0.707)),
            Pose(position=Point(x=-0.159, y=0.501, z=0.415), orientation=Quaternion(x=0.029, y=0.997, z=0.045, w=0.033)),
            Pose(position=Point(x=-0.806, y=0.010, z=0.182), orientation=Quaternion(x=-0.684, y=0.726, z=0.050, w=0.008))
        ]
        self.ik_solution_waypoints = []
        self.waypoint_index = 0
        self.waypoint_names = ["P1", "P2", "P3"]

        # --- PD Controller Parameters ---
        self.kp_gain = 8.0
        self.kd_gain = 4.0
        self.angle_tolerance = 0.05 # Initial tolerance for staging move
        self.previous_error = np.zeros(len(JOINT_ORDER))

        # --- Initialization & Control Loop Start ---
        self.control_loop_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Controller node initialized.')

    def joint_states_callback(self, msg):
        '''
        Callback to update the current joint angles from the /joint_states topic.
        '''
        if self.joint_names_in_message is None: 
            self.joint_names_in_message = list(msg.name)
        
        ordered_angles = np.zeros(len(JOINT_ORDER))
        for i, name in enumerate(JOINT_ORDER):
            if name in self.joint_names_in_message:
                idx = self.joint_names_in_message.index(name)
                ordered_angles[i] = msg.position[idx]
        self.current_joint_angles = ordered_angles

    def calculate_all_ik_solutions(self):
        '''
        Iterates through waypoints, calling the IK service for each.
        '''
        self.get_logger().info('Calculating IK solutions for all waypoints...')
        
        for i, pose in enumerate(self.cartesian_waypoints):
            req = GetPositionIK.Request()
            req.ik_request.group_name = 'ur_manipulator'
            req.ik_request.pose_stamped.header.frame_id = 'base_link'
            req.ik_request.pose_stamped.pose = pose
            req.ik_request.avoid_collisions = True

            joint_state = JointState()
            joint_state.name = JOINT_ORDER
            # Seed IK with the previous solution for better results
            if i > 0 and len(self.ik_solution_waypoints) > 0:
                joint_state.position = self.ik_solution_waypoints[-1].tolist()
            else:
                joint_state.position = self.current_joint_angles.tolist()
            req.ik_request.robot_state.joint_state = joint_state
            
            future = self.ik_client.call_async(req)
            
            def process_ik_response(future_response, waypoint_idx):
                result = future_response.result()
                if result and result.error_code.val == result.error_code.SUCCESS:
                    solution_angles = result.solution.joint_state.position
                    ordered_solution = np.zeros(len(JOINT_ORDER))
                    sol_names = result.solution.joint_state.name
                    for i_sol, name in enumerate(JOINT_ORDER):
                         if name in sol_names:
                            idx = sol_names.index(name)
                            ordered_solution[i_sol] = solution_angles[idx]

                    self.ik_solution_waypoints.append(ordered_solution)
                    self.get_logger().info(f'IK Solution found for {self.waypoint_names[waypoint_idx]}')

                    # If all solutions are found, transition to the next state
                    if len(self.ik_solution_waypoints) == len(self.cartesian_waypoints):
                        self.get_logger().info('All IK solutions found. Proceeding to move.')
                        self.angle_tolerance = 0.04 # Tighten tolerance for competition waypoints
                        self.active_target_angles = self.ik_solution_waypoints[self.waypoint_index]
                        self.state = 'MOVING_TO_WAYPOINT'
                else:
                    error_code = "N/A" if not result else result.error_code.val
                    self.get_logger().error(f'IK failed for {self.waypoint_names[waypoint_idx]}. Error code: {error_code}. Shutting down.')
                    rclpy.shutdown()

            future.add_done_callback(lambda fut, idx=i: process_ik_response(fut, idx))

    def control_loop(self):
        '''
        The main control loop state machine. Handles moving to a joint staging pose,
        calculating IK, and then moving through the Cartesian waypoints.
        '''
        if self.current_joint_angles is None: 
            self.get_logger().info("Waiting for /joint_states...", throttle_duration_sec=2)
            return

        if self.state == 'INITIALIZING':
            self.get_logger().info("Moving to a joint staging pose first.")
            self.active_target_angles = self.staging_joint_angles
            self.state = 'MOVING_TO_JOINT_STAGING'
        
        elif self.state == 'MOVING_TO_JOINT_STAGING':
            error = self.active_target_angles - self.current_joint_angles
            error = np.arctan2(np.sin(error), np.cos(error))
            self.get_logger().info(f"Moving to Staging Pose... Error norm: {np.linalg.norm(error):.3f}", throttle_duration_sec=1)

            if np.linalg.norm(error) < self.angle_tolerance:
                self.get_logger().info("✅ Staging Pose reached. Now calculating IK solutions.")
                self.publisher.publish(Float64MultiArray(data=[0.0]*6))
                self.previous_error = np.zeros(len(JOINT_ORDER))
                self.state = 'CALCULATING_IK'
                # Use a one-shot timer to avoid blocking the control loop while calling the service
                self.create_timer(0.1, self.calculate_all_ik_solutions, oneshot=True) 
                return

            derivative = error - self.previous_error
            velocities = (self.kp_gain * error) + (self.kd_gain * derivative)
            velocities = np.clip(velocities, -2.0, 2.0)
            self.publisher.publish(Float64MultiArray(data=list(velocities)))
            self.previous_error = error

        elif self.state == 'CALCULATING_IK':
            # This is a transient state. We stop sending motor commands and wait for the
            # process_ik_response callback to change the state to 'MOVING_TO_WAYPOINT'.
            self.get_logger().info("...Waiting for IK solutions...", throttle_duration_sec=1)
            self.publisher.publish(Float64MultiArray(data=[0.0]*6))

        elif self.state == 'MOVING_TO_WAYPOINT':
            current_waypoint_name = self.waypoint_names[self.waypoint_index]
            error = self.active_target_angles - self.current_joint_angles
            error = np.arctan2(np.sin(error), np.cos(error))
            self.get_logger().info(f"Moving to {current_waypoint_name}... Error norm: {np.linalg.norm(error):.3f}", throttle_duration_sec=1)
            
            if np.linalg.norm(error) < self.angle_tolerance:
                self.get_logger().info(f"✅ Arrived at Waypoint {current_waypoint_name}! Holding for 1 second.")
                self.publisher.publish(Float64MultiArray(data=[0.0]*6))
                time.sleep(1.0)
                
                self.waypoint_index += 1
                self.previous_error = np.zeros(len(JOINT_ORDER))
                
                if self.waypoint_index >= len(self.ik_solution_waypoints):
                    self.state = 'DONE'
                else:
                    next_waypoint_name = self.waypoint_names[self.waypoint_index]
                    self.get_logger().info(f"Moving to next waypoint {next_waypoint_name}...")
                    self.active_target_angles = self.ik_solution_waypoints[self.waypoint_index]
                return

            derivative = error - self.previous_error
            velocities = (self.kp_gain * error) + (self.kd_gain * derivative)
            velocities = np.clip(velocities, -4.0, 4.0)
            self.publisher.publish(Float64MultiArray(data=list(velocities)))
            self.previous_error = error

        elif self.state == 'DONE':
            self.get_logger().info("🎉 Sequence complete!")
            self.publisher.publish(Float64MultiArray(data=[0.0]*6))
            self.destroy_timer(self.control_loop_timer)
            rclpy.shutdown()

def main(args=None):
    '''
    Initializes ROS 2, creates the controller node, and spins it until shutdown.
    '''
    rclpy.init(args=args)
    node = FinalArmController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()

if __name__ == '__main__':
    main()


# Chapter 2: Physical AI for Humanoid Robotics

## 2.1 Introduction to Physical AI in Humanoid Robotics

Physical AI represents the integration of artificial intelligence with physical systems, enabling robots to perceive, reason, and act in the real world. For humanoid robotics, Physical AI encompasses the algorithms and systems that allow human-like robots to understand their environment, plan movements, and interact with objects and humans safely and effectively.

Unlike traditional AI that operates in virtual environments, Physical AI must contend with real-world constraints such as physics, sensor noise, actuator limitations, and the need for real-time responses. Humanoid robots face additional challenges due to their complex multi-degree-of-freedom systems, balance requirements, and anthropomorphic design goals.

### Key Components of Physical AI in Humanoid Robotics

1. **Perception Systems**: Visual, auditory, tactile, and proprioceptive sensing
2. **Motion Planning**: Trajectory generation and path planning
3. **Control Systems**: Low-level motor control and high-level behavior control
4. **Learning Systems**: Adaptation and improvement through experience
5. **Interaction Systems**: Human-robot interaction and social understanding

## 2.2 Theoretical Foundations

### 2.2.1 Robot Kinematics and Dynamics

Humanoid robots operate in 3D space with complex kinematic chains. Understanding forward and inverse kinematics is crucial for controlling limb movements.

**Forward Kinematics**: Given joint angles, calculate the position and orientation of the end-effector.
**Inverse Kinematics**: Given desired end-effector pose, calculate required joint angles.

For humanoid systems, we often deal with redundant systems where multiple joint configurations can achieve the same end-effector pose. This redundancy can be exploited for secondary objectives like obstacle avoidance or energy efficiency.

### 2.2.2 Control Theory in Physical AI

Control systems in humanoid robotics must handle:
- **Feedback Control**: Real-time adjustments based on sensor data
- **Feedforward Control**: Predictive control based on desired trajectories
- **Hierarchical Control**: Multi-level control architecture from high-level goals to low-level joint commands

### 2.2.3 Sensor Fusion and State Estimation

Humanoid robots utilize multiple sensors to estimate their state (position, velocity, orientation). Common approaches include:
- Kalman Filters for linear systems
- Extended Kalman Filters for non-linear systems
- Particle Filters for multi-modal distributions
- Complementary filters for combining different sensor types

### 2.2.4 Motion Planning for Humanoid Robots

Motion planning in humanoid robotics involves:
- **Configuration Space**: The space of all possible robot configurations
- **Collision Avoidance**: Ensuring the robot doesn't collide with obstacles
- **Dynamic Balance**: Maintaining stability during movement
- **Human-like Motion**: Generating natural, anthropomorphic movements

## 2.3 Simple Examples

### Example 1: Balance Control with Inverted Pendulum Model

A simple representation of humanoid balance can be modeled as an inverted pendulum where the center of mass is balanced above the support point.

```python
#!/usr/bin/env python3
"""
Simple inverted pendulum model for humanoid balance control
This demonstrates basic PID control for maintaining balance
"""

import numpy as np
import time
from collections import deque

class BalanceController:
    def __init__(self, kp=10.0, ki=0.1, kd=2.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain  
        self.kd = kd  # Derivative gain
        
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.target_pose = 0.0  # Target balance position (upright)
        
    def compute_control(self, current_pose, dt):
        # Calculate error
        error = self.target_pose - current_pose
        
        # Update integral term
        self.integral_error += error * dt
        
        # Calculate derivative term
        derivative = (error - self.previous_error) / dt
        
        # Compute PID output
        output = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        self.previous_error = error
        return output

class InvertedPendulum:
    def __init__(self, length=1.0, mass=1.0, g=9.81):
        self.length = length  # Length of pendulum
        self.mass = mass      # Mass of pendulum
        self.g = g            # Gravity
        self.angle = 0.0      # Current angle (radians)
        self.angular_velocity = 0.0  # Angular velocity (rad/s)
        
    def update(self, control_input, dt):
        # Calculate angular acceleration based on gravity and control input
        # For an inverted pendulum: angular_acc = (g/L) * sin(angle) + control_input
        angular_acc = (self.g / self.length) * np.sin(self.angle) + control_input
        
        # Update angular velocity and angle
        self.angular_velocity += angular_acc * dt
        self.angle += self.angular_velocity * dt
        
        return self.angle, self.angular_velocity

def simulate_balance():
    """Simulate balance control for an inverted pendulum"""
    controller = BalanceController()
    pendulum = InvertedPendulum(length=0.5, mass=10.0)
    
    dt = 0.01  # Time step (10ms)
    duration = 10.0  # Simulation duration (seconds)
    steps = int(duration / dt)
    
    # Simulate with some initial disturbance
    pendulum.angle = 0.1  # Initial angle (10 degrees)
    
    print("Time (s)\tAngle (rad)\tControl Output")
    print("--------\t-----------\t--------------")
    
    for i in range(steps):
        current_time = i * dt
        
        # Get control output based on current angle
        control_output = controller.compute_control(pendulum.angle, dt)
        
        # Update pendulum with control input
        angle, angular_velocity = pendulum.update(control_output, dt)
        
        if i % 100 == 0:  # Print every second
            print(f"{current_time:.2f}\t\t{angle:.4f}\t\t{control_output:.4f}")
        
        time.sleep(0.001)  # Slow down simulation for readability

if __name__ == "__main__":
    simulate_balance()
```

### Example 2: Basic Path Planning with Potential Fields

```python
#!/usr/bin/env python3
"""
Basic path planning using artificial potential fields
Demonstrates obstacle avoidance and goal seeking behavior
"""

import numpy as np
import matplotlib.pyplot as plt

class PotentialFieldPlanner:
    def __init__(self, goal, obstacles, robot_pos, attractive_gain=1.0, repulsive_gain=1.0, obstacle_radius=1.0):
        self.goal = np.array(goal)
        self.obstacles = [np.array(obs) for obs in obstacles]
        self.robot_pos = np.array(robot_pos)
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.obstacle_radius = obstacle_radius
        
    def attractive_force(self):
        """Calculate attractive force towards goal"""
        direction = self.goal - self.robot_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:  # Near goal
            return np.array([0.0, 0.0])
        
        # Linear attractive force
        force = self.attractive_gain * direction
        return force
    
    def repulsive_force(self):
        """Calculate repulsive force from obstacles"""
        total_force = np.array([0.0, 0.0])
        
        for obstacle in self.obstacles:
            direction = self.robot_pos - obstacle
            distance = np.linalg.norm(direction)
            
            if distance < self.obstacle_radius:
                # Calculate repulsive force
                if distance > 0.1:
                    force_magnitude = self.repulsive_gain * (1.0/distance - 1.0/self.obstacle_radius)
                    force_direction = direction / distance
                    force = force_magnitude * force_direction
                    total_force += force
                else:
                    # Avoid division by zero
                    force = np.array([0.0, 0.0])
                    total_force += force
        
        return total_force
    
    def total_force(self):
        """Calculate total force (attractive + repulsive)"""
        return self.attractive_force() + self.repulsive_force()
    
    def plan_step(self, step_size=0.1):
        """Plan one step of movement"""
        force = self.total_force()
        direction = force / (np.linalg.norm(force) + 1e-8)  # Normalize
        self.robot_pos += direction * step_size
        return self.robot_pos.copy()

def visualize_path_planning():
    """Visualize the potential field path planning"""
    goal = [8, 8]
    obstacles = [[3, 3], [5, 5], [6, 2]]
    start = [1, 1]
    
    planner = PotentialFieldPlanner(goal, obstacles, start)
    
    # Store path
    path = [start.copy()]
    max_steps = 500
    
    for i in range(max_steps):
        new_pos = planner.plan_step(step_size=0.1)
        path.append(new_pos.copy())
        
        # Check if reached goal (within tolerance)
        if np.linalg.norm(new_pos - goal) < 0.5:
            break
    
    # Convert to numpy arrays for plotting
    path = np.array(path)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), 0.5, color='red', alpha=0.5)
        plt.gca().add_patch(circle)
    
    # Plot path
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='Goal reached')
    plt.plot(goal[0], goal[1], 'rs', markersize=12, label='Target')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Potential Field Path Planning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    print(f"Path completed in {len(path)} steps")
    print(f"Final position: {path[-1]}")
    print(f"Distance to goal: {np.linalg.norm(path[-1] - goal):.2f}")

if __name__ == "__main__":
    visualize_path_planning()
```

## 2.4 ROS2 Python Implementation Examples

### Example 3: ROS2 Node for Joint Control

```python
#!/usr/bin/env python3
"""
ROS2 node for controlling humanoid robot joints
Implements trajectory execution with position, velocity, and acceleration
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np

class HumanoidJointController(Node):
    def __init__(self):
        super().__init__('humanoid_joint_controller')
        
        # Publisher for joint trajectory commands
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Joint names for humanoid (example: 24 DOF humanoid)
        self.joint_names = [
            'left_hip_roll', 'left_hip_yaw', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_roll', 'right_hip_yaw', 'right_hip_pitch', 
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'torso_yaw', 'torso_pitch', 'torso_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_yaw',
            'neck_yaw', 'neck_pitch'
        ]
        
        # Current joint positions
        self.current_positions = {name: 0.0 for name in self.joint_names}
        
        self.get_logger().info('Humanoid Joint Controller initialized')

    def joint_state_callback(self, msg):
        """Callback to update current joint positions"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Example: Send a simple trajectory for demonstration
        self.send_joint_trajectory()

    def send_joint_trajectory(self):
        """Send a joint trajectory message"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        # Create trajectory points
        points = []
        
        # Start position (current)
        start_point = JointTrajectoryPoint()
        start_point.positions = [self.current_positions[name] for name in self.joint_names]
        start_point.velocities = [0.0] * len(self.joint_names)
        start_point.accelerations = [0.0] * len(self.joint_names)
        start_point.time_from_start = Duration(sec=0, nanosec=0)
        points.append(start_point)
        
        # Mid position (example: lifted leg)
        mid_point = JointTrajectoryPoint()
        mid_positions = []
        for name in self.joint_names:
            if 'left_knee' in name:
                mid_positions.append(0.5)  # Lift knee
            elif 'left_hip_pitch' in name:
                mid_positions.append(0.2)  # Move hip forward
            else:
                mid_positions.append(self.current_positions[name])
        
        mid_point.positions = mid_positions
        mid_point.velocities = [0.0] * len(self.joint_names)
        mid_point.accelerations = [0.0] * len(self.joint_names)
        mid_point.time_from_start = Duration(sec=2, nanosec=0)
        points.append(mid_point)
        
        # End position (return to neutral)
        end_point = JointTrajectoryPoint()
        end_point.positions = [self.current_positions[name] for name in self.joint_names]
        end_point.velocities = [0.0] * len(self.joint_names)
        end_point.accelerations = [0.0] * len(self.joint_names)
        end_point.time_from_start = Duration(sec=4, nanosec=0)
        points.append(end_point)
        
        trajectory_msg.points = points
        self.trajectory_pub.publish(trajectory_msg)
        
        self.get_logger().info('Sent joint trajectory command')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidJointController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 4: ROS2 Perception Node for Object Detection

```python
#!/usr/bin/env python3
"""
ROS2 perception node for object detection and recognition
Uses camera data to detect objects and estimate their positions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String

class HumanoidPerceptionNode(Node):
    def __init__(self):
        super().__init__('humanoid_perception')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.object_pub = self.create_publisher(PointStamped, '/detected_objects', 10)
        self.status_pub = self.create_publisher(String, '/perception_status', 10)
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_info_received = False
        
        # Object detection parameters
        self.object_classifier = cv2.CascadeClassifier()
        
        # Initialize Haar cascade for simple object detection
        # In practice, you'd use a more sophisticated model
        self.get_logger().info('Humanoid Perception Node initialized')

    def camera_info_callback(self, msg):
        """Update camera parameters"""
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('Camera parameters updated')

    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
        
        # Convert to grayscale for object detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Simple circle detection as an example
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            min_dist=50,
            param1=50,
            param2=30,
            min_radius=10,
            max_radius=100
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Draw detected circles
                cv2.circle(cv_image, (x, y), r, (0, 255, 0), 2)
                
                # Publish object position as 3D point
                if self.camera_info_received:
                    object_3d = self.convert_2d_to_3d(x, y, depth=1.0)  # Assuming 1m depth
                    self.publish_object_position(object_3d)
                
                # Annotate image
                cv2.putText(cv_image, f'Object at ({x}, {y})', 
                           (x - 30, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Objects detected: {len(circles) if circles is not None else 0}"
        self.status_pub.publish(status_msg)
        
        # For visualization, we could publish the processed image
        # (In practice, you'd use an image publisher)

    def convert_2d_to_3d(self, x, y, depth=1.0):
        """Convert 2D image coordinates to 3D world coordinates"""
        if self.camera_matrix is None:
            return Point()
        
        # Convert pixel coordinates to normalized coordinates
        x_norm = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_norm = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # Convert to 3D world coordinates
        world_x = x_norm * depth
        world_y = y_norm * depth
        world_z = depth
        
        return Point(x=world_x, y=world_y, z=world_z)

    def publish_object_position(self, point):
        """Publish detected object position"""
        point_stamped = PointStamped()
        point_stamped.header = Header()
        point_stamped.header.stamp = self.get_clock().now().to_msg()
        point_stamped.header.frame_id = "camera_frame"
        point_stamped.point = point
        
        self.object_pub.publish(point_stamped)
        self.get_logger().info(f'Published object at: ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})')

def main(args=None):
    rclpy.init(args=args)
    perception_node = HumanoidPerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.5 Laboratory Exercises

### Lab Exercise 1: Balance Control Implementation

**Objective**: Implement a balance controller for a humanoid robot using PID control principles.

**Equipment Required**:
- Simulated humanoid robot (Gazebo + ROS2)
- Computer with ROS2 installed
- Basic programming environment

**Procedure**:
1. Set up the simulation environment with a humanoid robot model
2. Implement a PID controller for balance using the example code as a starting point
3. Test the controller with different gain values (P, I, D)
4. Add external disturbances to test robustness
5. Analyze the response characteristics and stability

**Expected Outcomes**:
- Stable balance control of the simulated humanoid
- Understanding of PID tuning for physical systems
- Experience with sensor feedback and control loops

### Lab Exercise 2: Inverse Kinematics for Arm Control

**Objective**: Implement inverse kinematics to control the humanoid's arm to reach specified positions.

**Theory Background**:
Inverse kinematics (IK) solves for joint angles required to achieve a desired end-effector position. For redundant systems, optimization criteria can be added.

**Implementation Steps**:
1. Define the kinematic chain for the humanoid arm
2. Implement Jacobian-based inverse kinematics
3. Add constraints for joint limits and collision avoidance
4. Test with various target positions

```python
#!/usr/bin/env python3
"""
Inverse Kinematics Implementation Lab Exercise
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidArmIK:
    def __init__(self, link_lengths=None):
        if link_lengths is None:
            # Example: simple 2-link arm (upper arm, lower arm)
            self.link_lengths = [0.3, 0.3]  # meters
        else:
            self.link_lengths = link_lengths
        
        # Initial joint angles
        self.joint_angles = [0.0, 0.0]
        
    def jacobian(self):
        """Calculate the Jacobian matrix for the 2-DOF arm"""
        theta1, theta2 = self.joint_angles
        l1, l2 = self.link_lengths
        
        # Calculate Jacobian elements
        J = np.array([
            [-l1*np.sin(theta1) - l2*np.sin(theta1 + theta2), -l2*np.sin(theta1 + theta2)],
            [l1*np.cos(theta1) + l2*np.cos(theta1 + theta2), l2*np.cos(theta1 + theta2)]
        ])
        
        return J
    
    def forward_kinematics(self):
        """Calculate end-effector position from joint angles"""
        theta1, theta2 = self.joint_angles
        l1, l2 = self.link_lengths
        
        x = l1*np.cos(theta1) + l2*np.cos(theta1 + theta2)
        y = l1*np.sin(theta1) + l2*np.sin(theta1 + theta2)
        
        return np.array([x, y])
    
    def solve_ik(self, target_pos, max_iterations=100, tolerance=1e-4):
        """Solve inverse kinematics using Jacobian transpose method"""
        for i in range(max_iterations):
            current_pos = self.forward_kinematics()
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tolerance:
                print(f"IK converged in {i+1} iterations")
                return True
            
            # Calculate Jacobian
            J = self.jacobian()
            
            # Update joint angles using Jacobian transpose
            # J^T * error gives joint velocity in direction of error
            delta_theta = 0.01 * J.T @ error
            
            # Apply joint limits (example: +/- 90 degrees)
            for j in range(len(self.joint_angles)):
                new_angle = self.joint_angles[j] + delta_theta[j]
                # Limit to +/- 90 degrees
                self.joint_angles[j] = np.clip(new_angle, -np.pi/2, np.pi/2)
        
        print(f"IK did not converge after {max_iterations} iterations")
        return False

def test_ik():
    """Test the inverse kinematics implementation"""
    ik_solver = HumanoidArmIK()
    
    # Test positions
    test_positions = [
        [0.4, 0.2],  # Reach forward
        [0.2, 0.4],  # Reach up
        [0.1, 0.1],  # Reach down
    ]
    
    for target in test_positions:
        print(f"\nAttempting to reach: {target}")
        
        # Reset to initial position
        ik_solver.joint_angles = [0.0, 0.0]
        
        success = ik_solver.solve_ik(np.array(target))
        
        if success:
            final_pos = ik_solver.forward_kinematics()
            print(f"Final joint angles: {ik_solver.joint_angles}")
            print(f"Target: {target}, Achieved: {final_pos}")
            print(f"Error: {np.linalg.norm(np.array(target) - final_pos):.4f}")
        else:
            print("IK solution failed")

if __name__ == "__main__":
    test_ik()
```

### Lab Exercise 3: Path Planning and Navigation

**Objective**: Implement and test a navigation system for humanoid robot path planning in an environment with obstacles.

**Theory Background**:
Path planning for humanoid robots must consider:
- Kinematic constraints
- Dynamic balance during movement
- Obstacle avoidance
- Energy efficiency

**Implementation Steps**:
1. Implement A* or D* path planning algorithm
2. Add humanoid-specific constraints (step size, balance)
3. Integrate with ROS2 navigation stack
4. Test in simulation with various obstacle configurations

### Lab Exercise 4: Sensor Fusion for State Estimation

**Objective**: Combine data from multiple sensors (IMU, encoders, cameras) to estimate the humanoid's state accurately.

**Theory Background**:
Sensor fusion combines complementary information from multiple sensors to improve estimation accuracy and reliability.

**Implementation Steps**:
1. Implement an Extended Kalman Filter (EKF)
2. Fuse IMU, encoder, and camera data
3. Test with simulated sensor noise
4. Compare fused estimate to ground truth

```python
#!/usr/bin/env python3
"""
Extended Kalman Filter for Humanoid State Estimation
"""

import numpy as np

class HumanoidEKF:
    def __init__(self):
        # State vector: [x, y, theta, vx, vy, omega]
        # x, y: position
        # theta: orientation
        # vx, vy: linear velocity
        # omega: angular velocity
        
        self.state_dim = 6
        self.state = np.zeros(self.state_dim)
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise (for different sensors)
        self.R_imu = np.diag([0.01, 0.01, 0.001])  # [ax, ay, omega]
        self.R_encoders = np.diag([0.05, 0.05])     # [vx, vy]
        self.R_vision = np.diag([0.1, 0.1])         # [x, y]
    
    def predict(self, dt):
        """Prediction step of EKF"""
        # State transition model (simplified - assume constant velocity model)
        F = np.eye(self.state_dim)
        
        # Add dynamics based on current state
        theta = self.state[2]
        vx = self.state[3]
        vy = self.state[4]
        
        # Update position based on velocity
        self.state[0] += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        self.state[1] += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        self.state[2] += self.state[5] * dt  # Update orientation
        
        # Jacobian of state transition
        F[0, 3] = np.cos(theta) * dt
        F[0, 4] = -np.sin(theta) * dt
        F[0, 5] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt
        
        F[1, 3] = np.sin(theta) * dt
        F[1, 4] = np.cos(theta) * dt
        F[1, 5] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        
        F[2, 5] = dt
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_with_imu(self, measurement):
        """Update with IMU measurement [ax, ay, omega]"""
        # Measurement model: directly measure acceleration and angular velocity
        H = np.zeros((3, self.state_dim))
        H[2, 5] = 1  # Measure angular velocity
        
        # Expected measurement (simplified model)
        expected = np.array([0, 0, self.state[5]])
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_with_encoders(self, measurement):
        """Update with encoder measurement [vx, vy]"""
        H = np.zeros((2, self.state_dim))
        H[0, 3] = 1  # Measure vx
        H[1, 4] = 1  # Measure vy
        
        # Expected measurement
        expected = self.state[3:5]
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_encoders
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_with_vision(self, measurement):
        """Update with vision measurement [x, y]"""
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1  # Measure x position
        H[1, 1] = 1  # Measure y position
        
        # Expected measurement
        expected = self.state[0:2]
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_vision
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

def simulate_sensor_fusion():
    """Simulate sensor fusion with multiple sensor types"""
    ekf = HumanoidEKF()
    
    dt = 0.1
    duration = 10.0
    steps = int(duration / dt)
    
    print("Time\tX\tY\tTheta\tVx\tVy\tOmega")
    print("-" * 50)
    
    for i in range(steps):
        current_time = i * dt
        
        # Predict step
        ekf.predict(dt)
        
        # Simulate sensor measurements with noise
        # Ground truth (for comparison)
        true_pos = [0.5 * current_time, 0.3 * current_time]
        true_vel = [0.5, 0.3]
        true_omega = 0.1
        
        # Add noise to measurements
        imu_measurement = np.array([
            np.random.normal(0, 0.01),  # ax
            np.random.normal(0, 0.01),  # ay
            np.random.normal(true_omega, 0.001)  # omega
        ])
        
        encoder_measurement = np.array([
            np.random.normal(true_vel[0], 0.02),  # vx
            np.random.normal(true_vel[1], 0.02)   # vy
        ])
        
        vision_measurement = np.array([
            np.random.normal(true_pos[0], 0.05),  # x
            np.random.normal(true_pos[1], 0.05)   # y
        ])
        
        # Update with different sensors at different rates
        if i % 2 == 0:  # Update with IMU every other step
            ekf.update_with_imu(imu_measurement)
        
        if i % 5 == 0:  # Update with encoders every 5 steps
            ekf.update_with_encoders(encoder_measurement)
        
        if i % 10 == 0:  # Update with vision every 10 steps
            ekf.update_with_vision(vision_measurement)
        
        if i % 10 == 0:  # Print state every 10 steps
            print(f"{current_time:.1f}\t{ekf.state[0]:.2f}\t{ekf.state[1]:.2f}\t"
                  f"{ekf.state[2]:.3f}\t{ekf.state[3]:.2f}\t{ekf.state[4]:.2f}\t{ekf.state[5]:.3f}")

if __name__ == "__main__":
    simulate_sensor_fusion()
```

## 2.6 Summary

Physical AI for humanoid robotics integrates multiple complex systems including perception, control, planning, and learning. The key challenges include real-time processing, handling of uncertainty, and ensuring safety and stability. This chapter has covered the theoretical foundations, practical examples, and laboratory exercises that form the basis for developing advanced humanoid robotic systems.

The examples provided demonstrate how to implement balance control, path planning, joint control, and sensor fusion using ROS2 and Python. These concepts form the foundation for building more sophisticated humanoid robotic applications.

In the next chapter, we will explore advanced topics in machine learning for humanoid robotics, including reinforcement learning, imitation learning, and adaptive control systems.

## 2.7 Discussion Questions

1. What are the main differences between physical AI and traditional AI approaches?
2. How do sensor fusion techniques improve the performance of humanoid robots?
3. What are the challenges of implementing real-time control in humanoid robotics?
4. How can learning algorithms be integrated with traditional control methods in humanoid robots?
5. Discuss the safety considerations that arise when humanoid robots operate in human environments.
# Chapter 3: Visualization and Animation for Humanoid Robotics AI

## 3.1 Introduction to Visualization and Animation in Humanoid Robotics

Visualization and animation play crucial roles in humanoid robotics, serving multiple purposes:
- Development and debugging of robotic systems
- Demonstration of capabilities to stakeholders
- Educational tools for understanding robot behavior
- Human-robot interaction enhancement
- Validation of AI decision-making processes

Effective visualization helps bridge the gap between abstract AI algorithms and tangible robot behaviors, making complex systems more understandable and accessible.

### Types of Visualization in Humanoid Robotics

1. **Simulation Environments**: Physics-based worlds for testing and validation
2. **Robot State Visualization**: Display of joint angles, sensor readings, and internal states
3. **AI Decision Visualization**: Showcasing how AI algorithms make decisions
4. **Behavior Animation**: Smooth transitions between robot poses and actions
5. **Data Visualization**: Graphical representation of sensor data and performance metrics

## 3.2 Theoretical Foundations

### 3.2.1 3D Transformation Mathematics

Understanding 3D transformations is essential for animating humanoid robots:
- **Translation**: Moving objects in 3D space
- **Rotation**: Orienting objects using rotation matrices or quaternions
- **Scaling**: Changing object dimensions
- **Homogeneous Coordinates**: Representing transformations as 4x4 matrices

### 3.2.2 Kinematic Chain Visualization

Humanoid robots consist of multiple connected rigid bodies forming kinematic chains. Visualizing these chains requires:
- Forward kinematics computation for end-effector positioning
- Joint coordinate frame representation
- Link geometry visualization
- Collision geometry display

### 3.2.3 Animation Principles

Animation in robotics follows key principles:
- **Smooth Transitions**: Continuous motion without abrupt changes
- **Realistic Timing**: Natural acceleration and deceleration curves
- **Balance Preservation**: Maintaining center of mass within support polygon
- **Physical Plausibility**: Movements respecting physical constraints

## 3.3 Visualization Frameworks and Tools

### 3.3.1 RViz: ROS Visualization Tool

RViz is the primary visualization tool for ROS-based humanoid robotics projects, offering:
- Real-time robot state visualization
- Sensor data display (laser scans, images, point clouds)
- Custom plugin development
- TF tree visualization

### 3.3.2 Gazebo: Physics Simulation Environment

Gazebo provides:
- Accurate physics simulation
- Sensor simulation
- Realistic environments
- Integration with ROS controllers

### 3.3.3 Blender: 3D Modeling and Animation

Blender can be used for:
- High-quality robot modeling
- Complex animations
- Rendered demonstrations
- Video creation for presentations

## 3.4 Simple Examples

### Example 1: Basic Robot State Visualization

```python
#!/usr/bin/env python3
"""
Basic robot state visualization using matplotlib
Shows joint positions and robot configuration in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

class HumanoidVisualizer:
    def __init__(self):
        # Define humanoid skeleton structure
        # Simplified for visualization: 6 DOF legs, 6 DOF arms, 3 DOF torso, 3 DOF head
        self.joint_names = [
            'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 
            'right_hip_pitch', 'right_knee', 'right_ankle_pitch',
            'left_shoulder_pitch', 'left_elbow', 'left_wrist_pitch',
            'right_shoulder_pitch', 'right_elbow', 'right_wrist_pitch',
            'torso_pitch', 'torso_yaw', 'torso_roll',
            'neck_pitch', 'neck_yaw'
        ]
        
        # Initialize with neutral pose
        self.joint_angles = {name: 0.0 for name in self.joint_names}
        
        # Define kinematic chain for visualization
        self.links = [
            # Torso links
            ('hip_center', 'torso', 0.5),
            ('torso', 'head', 0.25),
            
            # Left leg
            ('hip_center', 'left_hip', 0.05),
            ('left_hip', 'left_knee', 0.4),
            ('left_knee', 'left_ankle', 0.4),
            ('left_ankle', 'left_foot', 0.1),
            
            # Right leg
            ('hip_center', 'right_hip', -0.05),
            ('right_hip', 'right_knee', 0.4),
            ('right_knee', 'right_ankle', 0.4),
            ('right_ankle', 'right_foot', 0.1),
            
            # Left arm
            ('torso', 'left_shoulder', 0.15),
            ('left_shoulder', 'left_elbow', 0.3),
            ('left_elbow', 'left_wrist', 0.3),
            ('left_wrist', 'left_hand', 0.1),
            
            # Right arm
            ('torso', 'right_shoulder', -0.15),
            ('right_shoulder', 'right_elbow', 0.3),
            ('right_elbow', 'right_wrist', 0.3),
            ('right_wrist', 'right_hand', 0.1),
        ]
        
        # Joint positions (relative to parent)
        self.joint_positions = {
            'hip_center': (0, 0, 0),
            'torso': (0, 0, 0.5),
            'head': (0, 0, 0.75),
            'left_hip': (-0.1, 0, 0),
            'left_knee': (-0.1, 0, -0.4),
            'left_ankle': (-0.1, 0, -0.8),
            'left_foot': (-0.1, 0, -0.9),
            'right_hip': (0.1, 0, 0),
            'right_knee': (0.1, 0, -0.4),
            'right_ankle': (0.1, 0, -0.8),
            'right_foot': (0.1, 0, -0.9),
            'left_shoulder': (-0.15, 0, 0.5),
            'left_elbow': (-0.45, 0, 0.5),
            'left_wrist': (-0.75, 0, 0.5),
            'left_hand': (-0.85, 0, 0.5),
            'right_shoulder': (0.15, 0, 0.5),
            'right_elbow': (0.45, 0, 0.5),
            'right_wrist': (0.75, 0, 0.5),
            'right_hand': (0.85, 0, 0.5),
        }
        
    def calculate_forward_kinematics(self):
        """Calculate 3D positions of all body parts"""
        # For simplicity, using static positions
        # In a real implementation, this would use actual FK calculations
        positions = {}
        
        for joint, pos in self.joint_positions.items():
            positions[joint] = np.array(pos)
        
        return positions
    
    def draw_humanoid(self, ax, positions):
        """Draw the humanoid skeleton on the given axis"""
        ax.clear()
        
        # Draw links between joints
        for start_joint, end_joint, length in self.links:
            if start_joint in positions and end_joint in positions:
                start_pos = positions[start_joint]
                end_pos = positions[end_joint]
                
                # Draw line between joints
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 
                        'b-', linewidth=2)
        
        # Draw joints as spheres
        for joint_name, pos in positions.items():
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=50)
            ax.text(pos[0], pos[1], pos[2], joint_name.split('_')[0][:4], fontsize=8)
        
        # Set axis properties
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Humanoid Robot Visualization')

def animate_humanoid():
    """Animate basic humanoid movements"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    visualizer = HumanoidVisualizer()
    
    def animate(frame):
        # Simple animation: wave left arm
        visualizer.joint_angles['left_shoulder_pitch'] = 0.5 * np.sin(frame * 0.1)
        visualizer.joint_angles['left_elbow'] = 0.3 * np.sin(frame * 0.15)
        
        # Jumping motion
        offset = 0.1 * np.sin(frame * 0.2)
        # Adjust hip position for jumping effect
        visualizer.joint_positions['hip_center'] = (0, 0, offset)
        visualizer.joint_positions['torso'] = (0, 0, 0.5 + offset)
        visualizer.joint_positions['head'] = (0, 0, 0.75 + offset)
        
        # Update leg positions to maintain balance
        visualizer.joint_positions['left_knee'] = (-0.1, 0, -0.4 + offset)
        visualizer.joint_positions['left_ankle'] = (-0.1, 0, -0.8 + offset)
        visualizer.joint_positions['left_foot'] = (-0.1, 0, -0.9 + offset)
        visualizer.joint_positions['right_knee'] = (0.1, 0, -0.4 + offset)
        visualizer.joint_positions['right_ankle'] = (0.1, 0, -0.8 + offset)
        visualizer.joint_positions['right_foot'] = (0.1, 0, -0.9 + offset)
        
        positions = visualizer.calculate_forward_kinematics()
        visualizer.draw_humanoid(ax, positions)
    
    ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_humanoid()
```

### Example 2: Interactive Joint Control Visualization

```python
#!/usr/bin/env python3
"""
Interactive joint control visualization using PyQt
Allows manual adjustment of joint angles and visualization of resulting pose
"""

import sys
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSlider, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for drawing robot"""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class JointControlWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_robot()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Canvas for robot visualization
        self.canvas = MplCanvas(self, width=5, height=5, dpi=100)
        layout.addWidget(self.canvas)
        
        # Sliders for joint control
        joint_layout = QHBoxLayout()
        
        # Left shoulder pitch
        left_shoulder_layout = QVBoxLayout()
        left_shoulder_layout.addWidget(QLabel("Left Shoulder"))
        self.left_shoulder_slider = QSlider(Qt.Horizontal)
        self.left_shoulder_slider.setRange(-90, 90)
        self.left_shoulder_slider.setValue(0)
        self.left_shoulder_slider.valueChanged.connect(lambda v: self.update_robot())
        left_shoulder_layout.addWidget(self.left_shoulder_slider)
        
        # Left elbow
        left_elbow_layout = QVBoxLayout()
        left_elbow_layout.addWidget(QLabel("Left Elbow"))
        self.left_elbow_slider = QSlider(Qt.Horizontal)
        self.left_elbow_slider.setRange(-90, 90)
        self.left_elbow_slider.setValue(0)
        self.left_elbow_slider.valueChanged.connect(lambda v: self.update_robot())
        left_elbow_layout.addWidget(self.left_elbow_slider)
        
        # Right shoulder pitch
        right_shoulder_layout = QVBoxLayout()
        right_shoulder_layout.addWidget(QLabel("Right Shoulder"))
        self.right_shoulder_slider = QSlider(Qt.Horizontal)
        self.right_shoulder_slider.setRange(-90, 90)
        self.right_shoulder_slider.setValue(0)
        self.right_shoulder_slider.valueChanged.connect(lambda v: self.update_robot())
        right_shoulder_layout.addWidget(self.right_shoulder_slider)
        
        # Right elbow
        right_elbow_layout = QVBoxLayout()
        right_elbow_layout.addWidget(QLabel("Right Elbow"))
        self.right_elbow_slider = QSlider(Qt.Horizontal)
        self.right_elbow_slider.setRange(-90, 90)
        self.right_elbow_slider.setValue(0)
        self.right_elbow_slider.valueChanged.connect(lambda v: self.update_robot())
        right_elbow_layout.addWidget(self.right_elbow_slider)
        
        joint_layout.addLayout(left_shoulder_layout)
        joint_layout.addLayout(left_elbow_layout)
        joint_layout.addLayout(right_shoulder_layout)
        joint_layout.addLayout(right_elbow_layout)
        
        layout.addLayout(joint_layout)
        
        self.setLayout(layout)
        
    def init_robot(self):
        """Initialize robot kinematic structure"""
        # Robot dimensions (in arbitrary units)
        self.body_height = 1.0
        self.arm_length = 0.5
        self.leg_length = 0.7
        
        # Initial joint angles (in degrees)
        self.joint_angles = {
            'left_shoulder': 0,
            'left_elbow': 0,
            'right_shoulder': 0,
            'right_elbow': 0
        }
        
        self.update_robot()
        
    def calculate_arm_positions(self, shoulder_angle, elbow_angle, side):
        """Calculate arm joint positions based on angles"""
        # Convert to radians
        shoulder_rad = math.radians(shoulder_angle)
        elbow_rad = math.radians(elbow_angle)
        
        # Shoulder position (left or right)
        if side == 'left':
            shoulder_x = -0.15
        else:
            shoulder_x = 0.15
        shoulder_y = self.body_height * 0.8  # Shoulders at 80% body height
        
        # Elbow position
        elbow_x = shoulder_x + self.arm_length * math.sin(shoulder_rad)
        elbow_y = shoulder_y + self.arm_length * math.cos(shoulder_rad)
        
        # Wrist position
        wrist_x = elbow_x + self.arm_length * math.sin(shoulder_rad + elbow_rad)
        wrist_y = elbow_y + self.arm_length * math.cos(shoulder_rad + elbow_rad)
        
        return (shoulder_x, shoulder_y), (elbow_x, elbow_y), (wrist_x, wrist_y)
        
    def update_robot(self):
        """Redraw the robot based on current joint angles"""
        # Update joint angles from sliders
        self.joint_angles['left_shoulder'] = self.left_shoulder_slider.value()
        self.joint_angles['left_elbow'] = self.left_elbow_slider.value()
        self.joint_angles['right_shoulder'] = self.right_shoulder_slider.value()
        self.joint_angles['right_elbow'] = self.right_elbow_slider.value()
        
        # Clear the plot
        self.canvas.axes.clear()
        
        # Draw body (simple rectangle)
        self.canvas.axes.add_patch(plt.Rectangle((-0.15, 0), 0.3, self.body_height, 
                                                fill=True, color='lightblue', 
                                                edgecolor='black'))
        
        # Calculate and draw left arm
        left_shoulder_pos, left_elbow_pos, left_wrist_pos = self.calculate_arm_positions(
            self.joint_angles['left_shoulder'], 
            self.joint_angles['left_elbow'], 
            'left'
        )
        
        # Left arm links
        self.canvas.axes.plot([left_shoulder_pos[0], left_elbow_pos[0]], 
                             [left_shoulder_pos[1], left_elbow_pos[1]], 
                             'k-', linewidth=3)
        self.canvas.axes.plot([left_elbow_pos[0], left_wrist_pos[0]], 
                             [left_elbow_pos[1], left_wrist_pos[1]], 
                             'k-', linewidth=2)
        
        # Left joints
        self.canvas.axes.scatter(left_shoulder_pos[0], left_shoulder_pos[1], 
                                c='red', s=100, zorder=5)
        self.canvas.axes.scatter(left_elbow_pos[0], left_elbow_pos[1], 
                                c='red', s=80, zorder=5)
        self.canvas.axes.scatter(left_wrist_pos[0], left_wrist_pos[1], 
                                c='red', s=60, zorder=5)
        
        # Calculate and draw right arm
        right_shoulder_pos, right_elbow_pos, right_wrist_pos = self.calculate_arm_positions(
            self.joint_angles['right_shoulder'], 
            self.joint_angles['right_elbow'], 
            'right'
        )
        
        # Right arm links
        self.canvas.axes.plot([right_shoulder_pos[0], right_elbow_pos[0]], 
                             [right_shoulder_pos[1], right_elbow_pos[1]], 
                             'k-', linewidth=3)
        self.canvas.axes.plot([right_elbow_pos[0], right_wrist_pos[0]], 
                             [right_elbow_pos[1], right_wrist_pos[1]], 
                             'k-', linewidth=2)
        
        # Right joints
        self.canvas.axes.scatter(right_shoulder_pos[0], right_shoulder_pos[1], 
                                c='red', s=100, zorder=5)
        self.canvas.axes.scatter(right_elbow_pos[0], right_elbow_pos[1], 
                                c='red', s=80, zorder=5)
        self.canvas.axes.scatter(right_wrist_pos[0], right_wrist_pos[1], 
                                c='red', s=60, zorder=5)
        
        # Draw head
        self.canvas.axes.add_patch(Circle((0, self.body_height + 0.1), 0.1, 
                                         fill=True, color='lightgray', 
                                         edgecolor='black'))
        
        # Draw legs (fixed for now)
        # Left leg
        self.canvas.axes.plot([-0.1, -0.1], [0, -self.leg_length], 
                             'k-', linewidth=3)
        self.canvas.axes.scatter(-0.1, -self.leg_length, c='red', s=100, zorder=5)
        
        # Right leg
        self.canvas.axes.plot([0.1, 0.1], [0, -self.leg_length], 
                             'k-', linewidth=3)
        self.canvas.axes.scatter(0.1, -self.leg_length, c='red', s=100, zorder=5)
        
        # Set equal aspect ratio and limits
        self.canvas.axes.set_aspect('equal')
        self.canvas.axes.set_xlim(-0.5, 0.5)
        self.canvas.axes.set_ylim(-0.8, 1.2)
        self.canvas.axes.set_title('Humanoid Robot Joint Control')
        
        # Redraw
        self.canvas.draw()

class JointControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Humanoid Robot Joint Control')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        self.joint_widget = JointControlWidget()
        layout.addWidget(self.joint_widget)

def main():
    app = QApplication(sys.argv)
    window = JointControlWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

### Example 3: AI Behavior Visualization

```python
#!/usr/bin/env python3
"""
Visualization of AI decision-making process in humanoid robotics
Shows how state machines and behavior trees guide robot actions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import random

class BehaviorVisualizer:
    def __init__(self):
        # Define robot states
        self.states = [
            'Idle', 'Walking', 'Balancing', 'Interacting', 'Avoiding Obstacle', 'Planning'
        ]
        
        # Define possible state transitions
        self.transitions = {
            'Idle': ['Walking', 'Interacting', 'Idle'],
            'Walking': ['Balancing', 'Avoiding Obstacle', 'Idle'],
            'Balancing': ['Walking', 'Idle'],
            'Interacting': ['Idle'],
            'Avoiding Obstacle': ['Walking', 'Planning'],
            'Planning': ['Walking']
        }
        
        # Current state
        self.current_state = 'Idle'
        self.state_history = [self.current_state]
        
        # AI decision confidence levels
        self.confidence_levels = {}
        for state in self.states:
            self.confidence_levels[state] = 0.0
        
        # Robot position and velocity
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # Heading angle
        self.velocity = 0.0
        
        # Environment obstacles
        self.obstacles = [(2, 1), (3, -1), (4, 1.5), (5, -0.5)]
        
        # Sensory inputs
        self.sensor_readings = {
            'front_distance': 2.0,
            'left_distance': 1.5,
            'right_distance': 1.8,
            'center_of_pressure_x': 0.0,
            'center_of_pressure_y': 0.0
        }
    
    def update_behavior(self):
        """Update robot behavior based on sensory inputs"""
        # Get probabilities for each state based on sensor readings
        probabilities = self.calculate_state_probabilities()
        
        # Select next state based on probabilities
        states = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Choose next state (with some randomness for realism)
        next_state = np.random.choice(states, p=probs)
        
        self.current_state = next_state
        self.state_history.append(next_state)
        
        # Update confidence levels
        for state in self.states:
            self.confidence_levels[state] = probabilities.get(state, 0.0)
        
        # Update robot position based on current state
        self.update_robot_motion()
        
        # Update sensor readings
        self.update_sensors()
    
    def calculate_state_probabilities(self):
        """Calculate probability of each state based on sensor inputs"""
        probabilities = {}
        
        # Baseline probabilities
        baseline = 0.1
        
        # Increase probability based on sensor conditions
        if self.current_state == 'Avoiding Obstacle':
            # High probability to stay in obstacle avoidance if front distance is low
            probabilities['Avoiding Obstacle'] = 0.8 if self.sensor_readings['front_distance'] < 0.5 else 0.1
            probabilities['Balancing'] = 0.1
            probabilities['Idle'] = 0.05
            probabilities['Walking'] = 0.05
        elif self.current_state == 'Balancing':
            # High probability to continue balancing if CoP is off-center
            if abs(self.sensor_readings['center_of_pressure_x']) > 0.1:
                probabilities['Balancing'] = 0.7
                probabilities['Idle'] = 0.2
                probabilities['Walking'] = 0.1
            else:
                probabilities['Idle'] = 0.4
                probabilities['Walking'] = 0.4
                probabilities['Balancing'] = 0.2
        elif self.current_state == 'Walking':
            # Probability based on path conditions
            if self.sensor_readings['front_distance'] < 0.8:
                probabilities['Avoiding Obstacle'] = 0.6
                probabilities['Planning'] = 0.3
                probabilities['Walking'] = 0.1
            else:
                probabilities['Walking'] = 0.7
                probabilities['Balancing'] = 0.2
                probabilities['Idle'] = 0.1
        else:  # Idle state
            if self.sensor_readings['front_distance'] > 1.0:
                probabilities['Walking'] = 0.6
                probabilities['Idle'] = 0.4
            else:
                probabilities['Idle'] = 0.6
                probabilities['Planning'] = 0.2
                probabilities['Avoiding Obstacle'] = 0.2
        
        # Fill in remaining probabilities
        remaining_prob = 1.0 - sum(probabilities.values())
        for state in self.states:
            if state not in probabilities:
                probabilities[state] = baseline * remaining_prob / (len(self.states) - len(probabilities))
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for state in probabilities:
                probabilities[state] /= total_prob
        
        return probabilities
    
    def update_robot_motion(self):
        """Update robot position based on current state"""
        dt = 0.1  # Time step
        
        if self.current_state == 'Walking':
            self.velocity = 0.5  # m/s
            self.robot_x += self.velocity * dt * np.cos(self.robot_theta)
            self.robot_y += self.velocity * dt * np.sin(self.robot_theta)
        elif self.current_state == 'Avoiding Obstacle':
            # Turn away from obstacles
            if self.sensor_readings['front_distance'] < 0.5:
                if self.sensor_readings['left_distance'] > self.sensor_readings['right_distance']:
                    self.robot_theta += 0.3 * dt  # Turn left
                else:
                    self.robot_theta -= 0.3 * dt  # Turn right
            self.velocity = 0.2
            self.robot_x += self.velocity * dt * np.cos(self.robot_theta)
            self.robot_y += self.velocity * dt * np.sin(self.robot_theta)
        elif self.current_state == 'Balancing':
            # Small corrective motions
            self.robot_x += 0.05 * np.sin(2 * np.pi * 0.5 * dt) * dt
            self.robot_y += 0.05 * np.cos(2 * np.pi * 0.5 * dt) * dt
        else:  # Idle
            self.velocity = 0.0
    
    def update_sensors(self):
        """Update sensor readings"""
        # Simulate sensor noise and dynamic obstacles
        self.sensor_readings['front_distance'] = max(0.1, self.sensor_readings['front_distance'] + 
                                                   np.random.normal(0, 0.05))
        self.sensor_readings['left_distance'] = max(0.1, self.sensor_readings['left_distance'] + 
                                                  np.random.normal(0, 0.05))
        self.sensor_readings['right_distance'] = max(0.1, self.sensor_readings['right_distance'] + 
                                                   np.random.normal(0, 0.05))
        
        # Update center of pressure (CoP) with some drift
        self.sensor_readings['center_of_pressure_x'] += np.random.normal(0, 0.01)
        self.sensor_readings['center_of_pressure_y'] += np.random.normal(0, 0.01)
        
        # Bound CoP values
        self.sensor_readings['center_of_pressure_x'] = np.clip(
            self.sensor_readings['center_of_pressure_x'], -0.15, 0.15
        )
        self.sensor_readings['center_of_pressure_y'] = np.clip(
            self.sensor_readings['center_of_pressure_y'], -0.15, 0.15
        )

def visualize_behavior():
    """Animate the behavior visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    visualizer = BehaviorVisualizer()
    
    # Initialize plots
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Robot Navigation Environment')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Plot obstacles
    for obs in visualizer.obstacles:
        circle = patches.Circle(obs, 0.2, color='red', alpha=0.5)
        ax1.add_patch(circle)
    
    # Robot patch
    robot_patch = patches.Circle((visualizer.robot_x, visualizer.robot_y), 0.1, 
                                color='blue', alpha=0.7)
    ax1.add_patch(robot_patch)
    
    # State probability bars
    ax2.set_xlabel('Confidence Level')
    ax2.set_ylabel('State')
    ax2.set_xlim(0, 1)
    bars = ax2.barh(range(len(visualizer.states)), [0]*len(visualizer.states))
    
    # State history plot
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('State')
    ax3.set_ylim(-0.5, len(visualizer.states) - 0.5)
    state_y_positions = {state: idx for idx, state in enumerate(visualizer.states)}
    
    # Sensor readings plot
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Distance (m)')
    ax4.set_ylim(0, 3)
    
    def animate(frame):
        # Update behavior
        visualizer.update_behavior()
        
        # Update robot position
        robot_patch.center = (visualizer.robot_x, visualizer.robot_y)
        
        # Update state probability bars
        probabilities = list(visualizer.confidence_levels.values())
        for bar, prob in zip(bars, probabilities):
            bar.set_width(prob)
        
        # Update state history plot
        ax3.clear()
        ax3.set_ylim(-0.5, len(visualizer.states) - 0.5)
        ax3.set_xlim(0, max(10, len(visualizer.state_history)))
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('State')
        
        for t, state in enumerate(visualizer.state_history[-20:]):  # Last 20 states
            y_pos = state_y_positions[state]
            ax3.plot(t + len(visualizer.state_history) - 20, y_pos, 'o', color='green', markersize=8)
        
        ax3.set_yticks(list(range(len(visualizer.states))))
        ax3.set_yticklabels(visualizer.states)
        ax3.set_title('State History')
        
        # Update sensor readings plot
        ax4.clear()
        ax4.set_ylim(0, 3)
        ax4.set_xlim(max(0, len(visualizer.state_history)-50), len(visualizer.state_history)+1)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Distance (m)')
        
        # Plot sensor readings
        ax4.plot(range(len(visualizer.state_history)), 
                [reading['front_distance'] for i in range(len(visualizer.state_history))], 
                label='Front Distance', color='blue')
        ax4.plot(range(len(visualizer.state_history)), 
                [reading['left_distance'] for i in range(len(visualizer.state_history))], 
                label='Left Distance', color='orange')
        ax4.plot(range(len(visualizer.state_history)), 
                [reading['right_distance'] for i in range(len(visualizer.state_history))], 
                label='Right Distance', color='purple')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Obstacle Threshold')
        ax4.legend()
        ax4.set_title('Sensor Readings Over Time')
        
        return robot_patch, bars
    
    ani = FuncAnimation(fig, animate, frames=200, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_behavior()
```

## 3.4 ROS2 Visualization Examples

### Example 4: RViz Visualization with Marker Arrays

```python
#!/usr/bin/env python3
"""
ROS2 node for publishing visualization markers for RViz
Shows robot skeleton, paths, and AI decision visualizations
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math
import numpy as np

class RobotVizNode(Node):
    def __init__(self):
        super().__init__('robot_visualization_node')
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        
        # Timer for updating visualization
        self.timer = self.create_timer(0.1, self.update_visualization)
        
        # Robot state variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.path_points = []
        
        # Initialize robot skeleton points
        self.initialize_robot_skeleton()
        
        self.get_logger().info('Robot Visualization Node Started')

    def initialize_robot_skeleton(self):
        """Define robot skeleton structure"""
        # Body parts dimensions
        self.torso_height = 0.8
        self.head_radius = 0.1
        self.arm_length = 0.5
        self.leg_length = 0.7
        
        # Define skeleton points relative to robot center
        self.skeleton_points = {
            'head_top': Point(x=0.0, y=0.0, z=self.torso_height + self.head_radius),
            'head_center': Point(x=0.0, y=0.0, z=self.torso_height),
            'torso_bottom': Point(x=0.0, y=0.0, z=0.0),
            'left_shoulder': Point(x=-0.15, y=0.0, z=self.torso_height * 0.8),
            'right_shoulder': Point(x=0.15, y=0.0, z=self.torso_height * 0.8),
            'left_elbow': Point(x=-0.4, y=0.0, z=self.torso_height * 0.8),
            'right_elbow': Point(x=0.4, y=0.0, z=self.torso_height * 0.8),
            'left_wrist': Point(x=-0.65, y=0.0, z=self.torso_height * 0.8),
            'right_wrist': Point(x=0.65, y=0.0, z=self.torso_height * 0.8),
            'left_hip': Point(x=-0.1, y=0.0, z=0.0),
            'right_hip': Point(x=0.1, y=0.0, z=0.0),
            'left_knee': Point(x=-0.1, y=0.0, z=-0.35),
            'right_knee': Point(x=0.1, y=0.0, z=-0.35),
            'left_ankle': Point(x=-0.1, y=0.0, z=-0.7),
            'right_ankle': Point(x=0.1, y=0.0, z=-0.7),
            'left_foot': Point(x=-0.1, y=0.0, z=-0.75),
            'right_foot': Point(x=0.1, y=0.0, z=-0.75)
        }
        
        # Define skeleton connections (links)
        self.skeleton_links = [
            # Head to torso
            ('head_center', 'head_top'),
            ('torso_bottom', 'head_center'),
            
            # Left arm
            ('head_center', 'left_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            
            # Right arm
            ('head_center', 'right_shoulder'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # Legs
            ('torso_bottom', 'left_hip'),
            ('torso_bottom', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle'),
            ('left_ankle', 'left_foot'),
            ('right_ankle', 'right_foot')
        ]

    def get_world_coordinates(self, point):
        """Transform local point to world coordinates considering robot pose"""
        cos_theta = math.cos(self.robot_theta)
        sin_theta = math.sin(self.robot_theta)
        
        # Rotate point
        rotated_x = point.x * cos_theta - point.y * sin_theta
        rotated_y = point.x * sin_theta + point.y * cos_theta
        
        # Translate to robot position
        world_x = self.robot_x + rotated_x
        world_y = self.robot_y + rotated_y
        world_z = point.z
        
        return Point(x=world_x, y=world_y, z=world_z)

    def create_skeleton_marker(self):
        """Create marker for robot skeleton"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_skeleton"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Set pose
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 0.03  # Line width
        
        # Set color (blue for skeleton)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        # Add points to connect
        for start_joint, end_joint in self.skeleton_links:
            start_local = self.skeleton_points[start_joint]
            end_local = self.skeleton_points[end_joint]
            
            start_world = self.get_world_coordinates(start_local)
            end_world = self.get_world_coordinates(end_local)
            
            marker.points.append(start_world)
            marker.points.append(end_world)
        
        return marker

    def create_joints_marker(self):
        """Create markers for robot joints"""
        marker_array = MarkerArray()
        
        id_counter = 1
        for joint_name, local_point in self.skeleton_points.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "robot_joints"
            marker.id = id_counter
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Transform to world coordinates
            world_point = self.get_world_coordinates(local_point)
            marker.pose.position = world_point
            marker.pose.orientation.w = 1.0
            
            # Set scale
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            # Set color based on joint type
            if 'head' in joint_name:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif 'shoulder' in joint_name or 'elbow' in joint_name or 'wrist' in joint_name:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:  # legs
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            marker.color.a = 0.9
            
            # Add text label
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "joint_labels"
            text_marker.id = id_counter + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position text slightly above joint
            text_marker.pose.position = world_point
            text_marker.pose.position.z += 0.08
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = joint_name.replace('_', '\n')
            text_marker.scale.z = 0.08
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
            
            id_counter += 1
        
        return marker_array

    def create_path_marker(self):
        """Create path marker for robot trajectory"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create path points
        for i in range(max(0, len(self.path_points) - 20), len(self.path_points)):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = self.path_points[i][0]
            pose.pose.position.y = self.path_points[i][1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        return path_msg

    def create_decision_markers(self):
        """Create markers for AI decisions and planning"""
        marker_array = MarkerArray()
        
        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "ai_goals"
        goal_marker.id = 200
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        
        # Simple goal at (5, 0, 0)
        goal_marker.pose.position.x = 5.0
        goal_marker.pose.position.y = 0.0
        goal_marker.pose.position.z = 0.1
        goal_marker.pose.orientation.w = 1.0
        
        goal_marker.scale.x = 0.3
        goal_marker.scale.y = 0.3
        goal_marker.scale.z = 0.3
        
        goal_marker.color.r = 0.0
        goal_marker.color.g = 1.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.7
        
        # Obstacle markers (simple simulation)
        for i, (obs_x, obs_y) in enumerate([(2, 1), (3, -1), (4, 1.5)]):
            obs_marker = Marker()
            obs_marker.header.frame_id = "map"
            obs_marker.header.stamp = self.get_clock().now().to_msg()
            obs_marker.ns = "obstacles"
            obs_marker.id = 300 + i
            obs_marker.type = Marker.CYLINDER
            obs_marker.action = Marker.ADD
            
            obs_marker.pose.position.x = obs_x
            obs_marker.pose.position.y = obs_y
            obs_marker.pose.position.z = 0.3
            obs_marker.pose.orientation.w = 1.0
            
            obs_marker.scale.x = 0.4
            obs_marker.scale.y = 0.4
            obs_marker.scale.z = 0.6
            
            obs_marker.color.r = 1.0
            obs_marker.color.g = 0.0
            obs_marker.color.b = 0.0
            obs_marker.color.a = 0.6
        
        marker_array.markers.extend([goal_marker, obs_marker])
        return marker_array

    def update_visualization(self):
        """Update all visualization markers"""
        # Update robot position (simple circular motion for demo)
        t = self.get_clock().now().nanoseconds / 1e9
        self.robot_x = 2.0 * math.cos(0.5 * t)
        self.robot_y = 2.0 * math.sin(0.5 * t)
        self.robot_theta = 0.5 * t  # Robot heading
        
        # Add to path
        self.path_points.append((self.robot_x, self.robot_y))
        if len(self.path_points) > 200:  # Limit path length
            self.path_points.pop(0)
        
        # Create and publish markers
        marker_array = MarkerArray()
        
        # Add skeleton marker
        skeleton_marker = self.create_skeleton_marker()
        marker_array.markers.append(skeleton_marker)
        
        # Add joint markers
        joint_markers = self.create_joints_marker()
        marker_array.markers.extend(joint_markers.markers)
        
        # Add decision markers
        decision_markers = self.create_decision_markers()
        marker_array.markers.extend(decision_markers.markers)
        
        self.marker_pub.publish(marker_array)
        
        # Publish path
        path_msg = self.create_path_marker()
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    viz_node = RobotVizNode()
    
    try:
        rclpy.spin(viz_node)
    except KeyboardInterrupt:
        pass
    finally:
        viz_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Example 5: Animation of Walking Pattern

```python
#!/usr/bin/env python3
"""
Animation of humanoid walking pattern
Demonstrates gait cycles and coordination between limbs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

class WalkingPatternVisualizer:
    def __init__(self):
        # Robot dimensions
        self.body_height = 1.0
        self.hip_width = 0.2
        self.thigh_length = 0.4
        self.shin_length = 0.4
        self.upper_arm_length = 0.3
        self.forearm_length = 0.3
        
        # Walking gait parameters
        self.stride_length = 0.6
        self.step_height = 0.1
        self.gait_cycle_time = 2.0  # seconds per step cycle
        
        # Initialize joint angles
        self.joint_angles = {
            'left_hip_pitch': 0.0,
            'left_knee': 0.0,
            'left_ankle': 0.0,
            'right_hip_pitch': 0.0,
            'right_knee': 0.0,
            'right_ankle': 0.0,
            'left_shoulder_pitch': 0.0,
            'left_elbow': 0.0,
            'right_shoulder_pitch': 0.0,
            'right_elbow': 0.0,
            'torso_pitch': 0.0
        }
    
    def calculate_leg_position(self, leg_side, phase, speed=0.5):
        """Calculate leg position based on gait phase"""
        # Gait phase: 0 to 1, where 0 is heel strike, 0.5 is mid-stance, 1 is toe-off
        if leg_side == 'left':
            # Left leg swings forward during first half of cycle
            swing_phase = phase if phase < 0.5 else 1.0 - phase
        else:
            # Right leg swings forward during second half of cycle
            swing_phase = phase - 0.5 if phase >= 0.5 else phase + 0.5
        
        # Hip movement (forward/backward)
        hip_offset_x = self.stride_length * (swing_phase - 0.5) * speed
        
        # Knee lift for swing phase
        knee_lift = self.step_height * math.sin(math.pi * swing_phase) * 2.0  # Amplified for clear visualization
        
        # Calculate joint angles using inverse kinematics approximation
        leg_length = self.thigh_length + self.shin_length
        
        # Simple sagittal plane gait
        if leg_side == 'left':
            hip_pitch = math.atan2(knee_lift, hip_offset_x + self.stride_length/2) if abs(knee_lift) > 0.01 else 0
            knee_angle = -math.pi/3 + knee_lift * 0.5  # Knee flexion during swing
            ankle_angle = -hip_pitch * 0.5  # Ankle compensation
        else:
            hip_pitch = math.atan2(-knee_lift, -(hip_offset_x + self.stride_length/2)) if abs(knee_lift) > 0.01 else 0
            knee_angle = math.pi/3 - knee_lift * 0.5  # Opposite knee bend
            ankle_angle = hip_pitch * 0.5  # Ankle compensation
        
        return hip_pitch, knee_angle, ankle_angle
    
    def calculate_arm_swing(self, phase, leg_side):
        """Calculate arm swing synchronized with leg movement"""
        # Arms swing opposite to legs
        if leg_side == 'left':
            arm_phase = phase + 0.5 if phase < 0.5 else phase - 0.5
        else:
            arm_phase = phase
        
        # Shoulder and elbow movement
        shoulder_swing = 0.2 * math.sin(2 * math.pi * arm_phase)
        elbow_swing = 0.15 * math.sin(2 * math.pi * arm_phase + math.pi)  # Opposite to shoulder
        
        return shoulder_swing, elbow_swing
    
    def calculate_forward_kinematics(self, side, hip_angle, knee_angle, ankle_angle):
        """Calculate position of foot given joint angles"""
        if side == 'left':
            hip_x, hip_y = -self.hip_width/2, self.body_height
        else:
            hip_x, hip_y = self.hip_width/2, self.body_height
        
        # Thigh endpoint
        thigh_x = hip_x + self.thigh_length * math.sin(hip_angle)
        thigh_y = hip_y - self.thigh_length * math.cos(hip_angle)
        
        # Shin endpoint (knee to ankle)
        knee_x = thigh_x + self.shin_length * math.sin(hip_angle + knee_angle)
        knee_y = thigh_y - self.shin_length * math.cos(hip_angle + knee_angle)
        
        # Foot endpoint (ankle to toe)
        foot_x = knee_x + 0.1 * math.sin(hip_angle + knee_angle + ankle_angle)
        foot_y = knee_y - 0.1 * math.cos(hip_angle + knee_angle + ankle_angle)
        
        return (hip_x, hip_y), (thigh_x, thigh_y), (knee_x, knee_y), (foot_x, foot_y)
    
    def draw_humanoid(self, ax, phase):
        """Draw the humanoid in walking position"""
        ax.clear()
        
        # Calculate leg positions
        left_hip_ang, left_knee_ang, left_ankle_ang = self.calculate_leg_position('left', phase)
        right_hip_ang, right_knee_ang, right_ankle_ang = self.calculate_leg_position('right', phase)
        
        # Calculate arm swings
        left_shoulder_ang, left_elbow_ang = self.calculate_arm_swing(phase, 'left')
        right_shoulder_ang, right_elbow_ang = self.calculate_arm_swing(phase, 'right')
        
        # Get leg positions using forward kinematics
        left_hip_pos, left_knee_pos, left_ankle_pos, left_foot_pos = self.calculate_forward_kinematics(
            'left', left_hip_ang, left_knee_ang, left_ankle_ang)
        right_hip_pos, right_knee_pos, right_ankle_pos, right_foot_pos = self.calculate_forward_kinematics(
            'right', right_hip_ang, right_knee_ang, right_ankle_ang)
        
        # Draw body (torso)
        torso_x = 0
        torso_y = self.body_height
        neck_x = 0
        neck_y = self.body_height + 0.2
        head_x = 0
        head_y = self.body_height + 0.3
        
        ax.plot([torso_x, neck_x], [torso_y, neck_y], 'k-', linewidth=4, label='Torso')
        ax.plot([neck_x - 0.05, neck_x + 0.05], [neck_y, neck_y], 'k-', linewidth=2)  # Shoulders
        ax.scatter(head_x, head_y, c='lightblue', s=100, zorder=5, label='Head')
        
        # Draw left leg
        ax.plot([left_hip_pos[0], left_knee_pos[0], left_ankle_pos[0], left_foot_pos[0]],
                [left_hip_pos[1], left_knee_pos[1], left_ankle_pos[1], left_foot_pos[1]], 
                'b-', linewidth=3, label='Left Leg')
        
        # Draw right leg
        ax.plot([right_hip_pos[0], right_knee_pos[0], right_ankle_pos[0], right_foot_pos[0]],
                [right_hip_pos[1], right_knee_pos[1], right_ankle_pos[1], right_foot_pos[1]], 
                'r-', linewidth=3, label='Right Leg')
        
        # Draw arms (simplified)
        # Left arm
        left_shoulder_x = -0.15
        left_elbow_x = left_shoulder_x + self.upper_arm_length * math.sin(left_shoulder_ang)
        left_elbow_y = neck_y - self.upper_arm_length * math.cos(left_shoulder_ang)
        left_wrist_x = left_elbow_x + self.forearm_length * math.sin(left_shoulder_ang + left_elbow_ang)
        left_wrist_y = left_elbow_y - self.forearm_length * math.cos(left_shoulder_ang + left_elbow_ang)
        
        ax.plot([left_shoulder_x, left_elbow_x, left_wrist_x],
                [neck_y, left_elbow_y, left_wrist_y],
                'g-', linewidth=2, label='Left Arm')
        
        # Right arm
        right_shoulder_x = 0.15
        right_elbow_x = right_shoulder_x + self.upper_arm_length * math.sin(right_shoulder_ang)
        right_elbow_y = neck_y - self.upper_arm_length * math.cos(right_shoulder_ang)
        right_wrist_x = right_elbow_x + self.forearm_length * math.sin(right_shoulder_ang + right_elbow_ang)
        right_wrist_y = right_elbow_y - self.forearm_length * math.cos(right_shoulder_ang + right_elbow_ang)
        
        ax.plot([right_shoulder_x, right_elbow_x, right_wrist_x],
                [neck_y, right_elbow_y, right_wrist_y],
                'g-', linewidth=2, label='Right Arm')
        
        # Mark joints
        ax.scatter(left_hip_pos[0], left_hip_pos[1], c='red', s=50, zorder=5)
        ax.scatter(left_knee_pos[0], left_knee_pos[1], c='red', s=50, zorder=5)
        ax.scatter(left_ankle_pos[0], left_ankle_pos[1], c='red', s=50, zorder=5)
        ax.scatter(left_foot_pos[0], left_foot_pos[1], c='red', s=50, zorder=5)
        
        ax.scatter(right_hip_pos[0], right_hip_pos[1], c='red', s=50, zorder=5)
        ax.scatter(right_knee_pos[0], right_knee_pos[1], c='red', s=50, zorder=5)
        ax.scatter(right_ankle_pos[0], right_ankle_pos[1], c='red', s=50, zorder=5)
        ax.scatter(right_foot_pos[0], right_foot_pos[1], c='red', s=50, zorder=5)
        
        ax.scatter([-0.15, 0.15], [neck_y, neck_y], c='red', s=50, zorder=5)  # Shoulders
        ax.scatter([left_elbow_x, right_elbow_x], [left_elbow_y, right_elbow_y], c='red', s=40, zorder=5)
        ax.scatter([left_wrist_x, right_wrist_x], [left_wrist_y, right_wrist_y], c='red', s=40, zorder=5)
        
        # Add gait phase indicator
        ax.text(0.02, 0.98, f'Gait Phase: {phase:.2f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set axis properties
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Humanoid Walking Animation')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

def animate_walking():
    """Create walking animation"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    walker = WalkingPatternVisualizer()
    
    def animate(frame):
        # Calculate gait phase (0 to 1)
        phase = (frame % 120) / 120.0  # 120 frames per gait cycle
        walker.draw_humanoid(ax, phase)
    
    ani = FuncAnimation(fig, animate, frames=240, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_walking()
```

## 3.5 Laboratory Exercises

### Lab Exercise 1: Robot State Visualization Dashboard

**Objective**: Create a comprehensive visualization dashboard that displays multiple aspects of robot state simultaneously.

**Components to Implement**:
1. Robot skeleton visualization in 3D space
2. Joint angle displays with gauges
3. Sensor readings visualization
4. Path planning and navigation display
5. AI decision confidence indicators

**Implementation Steps**:
1. Create a main display showing the robot's current pose
2. Add real-time joint angle displays using gauge widgets
3. Implement sensor fusion visualization showing multiple input sources
4. Create path planning visualization with cost maps
5. Add decision-making visualization showing state transitions

### Lab Exercise 2: Interactive Animation Editor

**Objective**: Develop an interactive tool for creating and editing robot animations.

**Features to Include**:
1. Timeline controls for animation sequences
2. Keyframe editor for major poses
3. Interpolation between keyframes
4. Preview playback functionality
5. Export to ROS trajectory format

**Implementation Steps**:
1. Design a timeline interface with frame-by-frame controls
2. Implement pose editor for defining key joint positions
3. Add interpolation algorithms (linear, cubic, Bézier)
4. Create preview window with real-time animation
5. Implement trajectory export functionality

### Lab Exercise 3: AI Behavior Tree Visualizer

**Objective**: Create a real-time visualizer for AI behavior trees and state machines.

**Components to Implement**:
1. Behavior tree graph visualization
2. Real-time node activation highlighting
3. Decision flow visualization
4. Performance metrics display
5. Debug information overlay

**Implementation Steps**:
1. Parse behavior tree structure from ROS parameters
2. Create graphical representation of tree nodes
3. Highlight active nodes during execution
4. Show transition arrows indicating decision flow
5. Add performance counters for each node

## 3.6 Advanced Visualization Techniques

### 3.6.1 Real-Time Rendering with OpenGL

For high-performance visualization, OpenGL can be used to render complex robot models and environments with high fidelity.

### 3.6.2 Augmented Reality Integration

AR technologies can overlay robot visualization onto real-world environments, enhancing teleoperation and debugging capabilities.

### 3.6.3 Data-Driven Visualization

Visualization systems that adapt based on incoming sensor data, showing relevant information dynamically as the robot operates.

## 3.7 Best Practices for Visualization

1. **Performance Optimization**: Keep visualization code efficient to avoid impacting robot control timing
2. **Modular Design**: Separate visualization concerns from control logic
3. **Scalability**: Ensure visualization scales appropriately with robot complexity
4. **User-Friendliness**: Design intuitive interfaces for various user expertise levels
5. **Robustness**: Handle visualization failures gracefully without affecting robot operation

## 3.8 Summary

Visualization and animation are essential tools for developing, debugging, and demonstrating humanoid robotics AI systems. This chapter has covered various techniques from basic 2D visualization to complex 3D simulation environments. The examples provided demonstrate how to visualize robot states, AI decision-making processes, and coordinated movements.

Effective visualization bridges the gap between abstract AI algorithms and physical robot behaviors, making complex systems more transparent and controllable. As humanoid robotics continues to advance, sophisticated visualization tools will become increasingly important for both developers and end-users.

## 3.9 Discussion Questions

1. What are the key challenges in visualizing multi-degree-of-freedom humanoid robots?
2. How can real-time performance be maintained while providing rich visualization?
3. What role does visualization play in debugging AI decision-making systems?
4. How might AR technologies enhance humanoid robot visualization?
5. What are the most important elements to visualize when demonstrating robot capabilities to non-experts?
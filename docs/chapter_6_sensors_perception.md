# Chapter 6: Sensors and Perception Systems for Humanoid Robotics

## 6.1 Introduction to Sensor Systems in Humanoid Robotics

Sensors form the foundation of autonomous behavior in humanoid robotics, enabling robots to perceive their environment, monitor their state, and interact safely with humans and objects. Unlike traditional industrial robots operating in structured environments, humanoid robots must navigate complex, dynamic environments while maintaining balance and safety.

The sensor suite of a humanoid robot typically includes:
- **Proprioceptive sensors**: Monitor internal state (joint angles, motor currents, IMU)
- **Exteroceptive sensors**: Perceive external environment (cameras, LIDAR, force sensors)
- **Cognitive sensors**: Assess interaction context (microphones, gesture recognition)

### Sensor Requirements for Humanoid Robots

Humanoid robots have unique sensor requirements due to their dynamic nature:
- **High-frequency sampling**: Balance control requires 1000Hz+ IMU data
- **Robustness**: Sensors must withstand impacts and vibrations
- **Low latency**: Real-time control demands minimal sensor delay
- **Redundancy**: Multiple sensors for critical functions
- **Human-safe operation**: Sensors must not pose risks to humans

## 6.2 Proprioceptive Sensors

### 6.2.1 Joint Position and Velocity Sensors

Joint encoders provide absolute or relative position feedback for each degree of freedom in the humanoid robot.

```python
#!/usr/bin/env python3
"""
Joint encoder interface for humanoid robotics
Demonstrates position and velocity feedback processing
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class JointState:
    """Data structure for joint state information"""
    name: str
    position: float  # Radians
    velocity: float  # Rad/s
    effort: float    # N·m
    timestamp: float

class JointEncoderArray:
    """Interface for managing multiple joint encoders"""
    def __init__(self, joint_names: List[str], encoder_resolution: int = 4096):
        self.joint_names = joint_names
        self.encoder_resolution = encoder_resolution  # Counts per revolution
        self.num_joints = len(joint_names)
        
        # Initialize with neutral position
        self.current_positions = [0.0] * self.num_joints
        self.last_positions = [0.0] * self.num_joints
        self.velocities = [0.0] * self.num_joints
        self.efforts = [0.0] * self.num_joints
        
        # For velocity calculation
        self.last_timestamps = [time.time()] * self.num_joints
        self.position_history = {name: deque(maxlen=10) for name in joint_names}
        
        # Velocity filtering
        self.velocity_filters = {name: SimpleLowPassFilter() for name in joint_names}
        
    def update_reading(self, joint_name: str, encoder_value: int):
        """Update with new encoder reading"""
        if joint_name not in self.position_history:
            raise ValueError(f"Unknown joint: {joint_name}")
        
        # Convert encoder value to radians
        position = self._encoder_to_radians(encoder_value)
        
        # Calculate velocity
        idx = self.joint_names.index(joint_name)
        current_time = time.time()
        dt = current_time - self.last_timestamps[idx]
        
        if dt > 0:
            raw_velocity = (position - self.current_positions[idx]) / dt
            # Apply low-pass filter
            filtered_velocity = self.velocity_filters[joint_name].filter(raw_velocity)
            
            self.velocities[idx] = filtered_velocity
            self.last_timestamps[idx] = current_time
        
        self.last_positions[idx] = self.current_positions[idx]
        self.current_positions[idx] = position
        self.position_history[joint_name].append(position)
    
    def _encoder_to_radians(self, encoder_value: int) -> float:
        """Convert encoder counts to radians"""
        # Assuming single revolution = encoder_resolution counts
        return (encoder_value % self.encoder_resolution) * 2 * np.pi / self.encoder_resolution
    
    def get_joint_state(self, joint_name: str) -> JointState:
        """Get current state of specified joint"""
        if joint_name not in self.position_history:
            raise ValueError(f"Unknown joint: {joint_name}")
        
        idx = self.joint_names.index(joint_name)
        return JointState(
            name=joint_name,
            position=self.current_positions[idx],
            velocity=self.velocities[idx],
            effort=self.efforts[idx],
            timestamp=time.time()
        )
    
    def get_all_states(self) -> List[JointState]:
        """Get state of all joints"""
        all_states = []
        current_time = time.time()
        
        for i, name in enumerate(self.joint_names):
            state = JointState(
                name=name,
                position=self.current_positions[i],
                velocity=self.velocities[i],
                effort=self.efforts[i],
                timestamp=current_time
            )
            all_states.append(state)
        
        return all_states

class SimpleLowPassFilter:
    """Simple low-pass filter for velocity calculation"""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.last_output = 0.0
        self.first_call = True
    
    def filter(self, input_value: float) -> float:
        if self.first_call:
            self.last_output = input_value
            self.first_call = False
            return input_value
        
        output = self.alpha * input_value + (1.0 - self.alpha) * self.last_output
        self.last_output = output
        return output

def simulate_joint_encoders():
    """Simulate joint encoder operation"""
    joint_names = [
        'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
        'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
    ]
    
    encoders = JointEncoderArray(joint_names)
    
    print("Simulating joint encoder updates...")
    
    for step in range(100):
        # Simulate encoder readings with some movement
        for i, joint_name in enumerate(joint_names):
            # Simulate sinusoidal motion
            angle = 0.2 * np.sin(0.1 * step + i * 0.5)
            encoder_counts = int((angle / (2 * np.pi)) * encoders.encoder_resolution)
            # Add noise
            encoder_counts += np.random.randint(-5, 5)
            
            encoders.update_reading(joint_name, encoder_counts)
        
        if step % 20 == 0:
            # Print state of first few joints
            for joint_name in joint_names[:3]:
                state = encoders.get_joint_state(joint_name)
                print(f"{state.name}: pos={state.position:.3f}, vel={state.velocity:.3f}, effort={state.effort:.3f}")
    
    print("Simulation completed.")

if __name__ == "__main__":
    simulate_joint_encoders()
```

### 6.2.2 Inertial Measurement Units (IMU)

IMUs are critical for balance control, providing orientation, angular velocity, and acceleration data.

```python
#!/usr/bin/env python3
"""
IMU interface and processing for humanoid robotics
Demonstrates orientation estimation and balance monitoring
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class IMUData:
    """IMU data structure"""
    timestamp: float
    linear_acceleration: np.ndarray  # [ax, ay, az]
    angular_velocity: np.ndarray     # [wx, wy, wz]
    orientation: np.ndarray          # [x, y, z, w] - quaternion
    magnetic_field: Optional[np.ndarray] = None

class IMUFilter:
    """IMU data processing and filtering"""
    def __init__(self, sample_rate: float = 1000.0):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Initial state
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.gravity_vector = np.array([0.0, 0.0, -9.81])
        
        # Bias estimation
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        
        # For drift correction
        self.accel_history = deque(maxlen=50)
        self.mag_history = deque(maxlen=50)
        
    def process_raw_data(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray = None) -> IMUData:
        """Process raw IMU data into filtered orientation"""
        current_time = time.time()
        
        # Apply bias correction
        corrected_gyro = gyro - self.gyro_bias
        corrected_accel = accel - self.accel_bias
        
        # Update orientation using gyro integration
        self._integrate_gyro(corrected_gyro)
        
        # Correct orientation using accelerometer
        self._correct_with_accelerometer(corrected_accel)
        
        # Store for bias estimation
        self.accel_history.append(corrected_accel)
        if mag is not None:
            self.mag_history.append(mag)
        
        # Update bias estimates periodically
        if len(self.accel_history) == 50:
            self._estimate_biases()
        
        return IMUData(
            timestamp=current_time,
            linear_acceleration=accel,
            angular_velocity=gyro,
            orientation=self.orientation.copy(),
            magnetic_field=mag
        )
    
    def _integrate_gyro(self, gyro: np.ndarray):
        """Integrate gyroscope data to update orientation"""
        # Convert angular velocity to quaternion derivative
        omega_quat = np.array([gyro[0], gyro[1], gyro[2], 0.0])
        omega_quat = self._quaternion_multiply(omega_quat, self.orientation) * 0.5
        
        # Integrate
        new_orientation = self.orientation + omega_quat * self.dt
        
        # Normalize quaternion
        self.orientation = new_orientation / np.linalg.norm(new_orientation)
    
    def _correct_with_accelerometer(self, accel: np.ndarray):
        """Use accelerometer to correct orientation drift"""
        # Normalize accelerometer reading
        accel_norm = accel / np.linalg.norm(accel)
        
        # Convert reference gravity vector to body frame
        gravity_body = self._rotate_vector_by_quaternion(self.gravity_vector, 
                                                        self._quaternion_conjugate(self.orientation))
        
        # Calculate correction quaternion
        correction_angle = np.arccos(np.clip(np.dot(gravity_body, accel_norm), -1.0, 1.0))
        
        if correction_angle > 0.01:  # Only correct if significant difference
            correction_axis = np.cross(gravity_body, accel_norm)
            correction_axis = correction_axis / np.linalg.norm(correction_axis)
            
            # Create correction quaternion
            correction_quat = np.array([
                correction_axis[0] * np.sin(correction_angle / 2),
                correction_axis[1] * np.sin(correction_angle / 2),
                correction_axis[2] * np.sin(correction_angle / 2),
                np.cos(correction_angle / 2)
            ])
            
            # Apply correction with small gain
            self.orientation = self._slerp(self.orientation, correction_quat, 0.01)
    
    def _estimate_biases(self):
        """Estimate and update bias values"""
        # Accelerometer bias: average should match gravity magnitude
        avg_accel = np.mean(self.accel_history, axis=0)
        gravity_magnitude = np.linalg.norm(avg_accel)
        
        if abs(gravity_magnitude - 9.81) < 0.5:  # Reasonable gravity reading
            self.accel_bias = avg_accel - np.array([0.0, 0.0, -9.81])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Get quaternion conjugate"""
        return np.array([-q[0], -q[1], -q[2], q[3]])
    
    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion"""
        v_quat = np.array([v[0], v[1], v[2], 0.0])
        q_conj = self._quaternion_conjugate(q)
        
        rotated = self._quaternion_multiply(self._quaternion_multiply(q, v_quat), q_conj)
        return rotated[:3]
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions"""
        dot = np.dot(q1, q2)
        
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result = s0 * q1 + s1 * q2
        return result / np.linalg.norm(result)
    
    def get_euler_angles(self) -> np.ndarray:
        """Convert quaternion to Euler angles [roll, pitch, yaw]"""
        w, x, y, z = self.orientation
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

def simulate_imu_processing():
    """Simulate IMU data processing"""
    imu_filter = IMUFilter(sample_rate=500.0)  # 500 Hz
    
    print("Simulating IMU processing...")
    
    for step in range(500):
        # Simulate IMU readings with some movement
        current_time = step / 500.0
        
        # Accelerometer: gravity + small movement
        accel = np.array([0.1 * np.sin(current_time), 
                         0.05 * np.cos(current_time), 
                         -9.81 + 0.1 * np.sin(2 * current_time)])
        
        # Gyro: rotation with drift
        gyro = np.array([0.01 * np.cos(current_time), 
                        0.005 * np.sin(current_time), 
                        0.002 * np.sin(0.5 * current_time)])
        
        # Add noise
        accel += np.random.normal(0, 0.01, 3)
        gyro += np.random.normal(0, 0.001, 3)
        
        # Process data
        imu_data = imu_filter.process_raw_data(accel, gyro)
        
        # Get Euler angles for balance monitoring
        euler = imu_filter.get_euler_angles()
        
        if step % 50 == 0:
            print(f"Time: {current_time:.2f}s")
            print(f"  Roll: {np.degrees(euler[0]):.2f}°, Pitch: {np.degrees(euler[1]):.2f}°, Yaw: {np.degrees(euler[2]):.2f}°")
            print(f"  Bias est: gyro={imu_filter.gyro_bias} accel={imu_filter.accel_bias}")
            print()

if __name__ == "__main__":
    from collections import deque  # Import here since it's used in the class
    simulate_imu_processing()
```

### 6.2.3 Force and Torque Sensors

Force/torque sensors enable precise interaction control and safety monitoring.

```python
#!/usr/bin/env python3
"""
Force/Torque sensor interface for humanoid robotics
Demonstrates contact detection and force control
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import threading

@dataclass
class ForceTorqueData:
    """Force/Torque sensor data"""
    timestamp: float
    forces: np.ndarray  # [fx, fy, fz]
    torques: np.ndarray  # [tx, ty, tz]
    contact_detected: bool

class ForceTorqueSensor:
    """Force/Torque sensor interface"""
    def __init__(self, sensor_location: str, force_threshold: float = 20.0):
        self.sensor_location = sensor_location
        self.force_threshold = force_threshold
        self.torque_threshold = 5.0
        
        # Current sensor reading
        self.forces = np.zeros(3)
        self.torques = np.zeros(3)
        self.contact_detected = False
        
        # Historical data for filtering
        self.force_history = np.zeros((10, 3))
        self.torque_history = np.zeros((10, 3))
        
        # Safety limits
        self.max_force = 100.0
        self.max_torque = 20.0
        
    def update_reading(self, raw_forces: np.ndarray, raw_torques: np.ndarray):
        """Update with new raw sensor readings"""
        # Apply basic filtering
        self._update_history(raw_forces, raw_torques)
        
        # Calculate filtered values
        self.forces = np.mean(self.force_history, axis=0)
        self.torques = np.mean(self.torque_history, axis=0)
        
        # Detect contact
        force_magnitude = np.linalg.norm(self.forces)
        torque_magnitude = np.linalg.norm(self.torques)
        
        self.contact_detected = (force_magnitude > self.force_threshold or 
                                torque_magnitude > self.torque_threshold)
        
        # Check safety limits
        if (np.any(np.abs(self.forces) > self.max_force) or 
            np.any(np.abs(self.torques) > self.max_torque)):
            print(f"WARNING: Force/Torque limits exceeded at {self.sensor_location}")
    
    def _update_history(self, new_forces: np.ndarray, new_torques: np.ndarray):
        """Update sensor history for filtering"""
        # Roll the history arrays and add new values
        self.force_history = np.roll(self.force_history, 1, axis=0)
        self.force_history[0] = new_forces
        
        self.torque_history = np.roll(self.torque_history, 1, axis=0)
        self.torque_history[0] = new_torques
    
    def get_data(self) -> ForceTorqueData:
        """Get current sensor data"""
        return ForceTorqueData(
            timestamp=time.time(),
            forces=self.forces.copy(),
            torques=self.torques.copy(),
            contact_detected=self.contact_detected
        )
    
    def is_safe_to_proceed(self) -> bool:
        """Check if forces are within safe limits"""
        force_magnitude = np.linalg.norm(self.forces)
        torque_magnitude = np.linalg.norm(self.torques)
        
        return (force_magnitude < self.max_force * 0.8 and 
                torque_magnitude < self.max_torque * 0.8)

class WholeBodyForceSensor:
    """Interface for multiple force/torque sensors"""
    def __init__(self):
        self.sensors = {}
        
        # Initialize sensors at key locations
        sensor_locations = [
            'left_foot', 'right_foot', 
            'left_hand', 'right_hand',
            'left_ankle', 'right_ankle'
        ]
        
        for location in sensor_locations:
            self.sensors[location] = ForceTorqueSensor(
                sensor_location=location
            )
        
        # Zero reference values
        self.zero_references = {loc: (np.zeros(3), np.zeros(3)) for loc in sensor_locations}
    
    def calibrate_sensor(self, location: str, num_samples: int = 100):
        """Calibrate sensor to account for static loads"""
        if location not in self.sensors:
            raise ValueError(f"Unknown sensor location: {location}")
        
        print(f"Calibrating {location} sensor...")
        
        cumulative_forces = np.zeros(3)
        cumulative_torques = np.zeros(3)
        
        for _ in range(num_samples):
            # In real implementation, would read actual sensor values
            # For simulation, we'll use nominal values
            base_forces = np.array([0, 0, -50.0]) if 'foot' in location else np.zeros(3)
            base_torques = np.zeros(3)
            
            cumulative_forces += base_forces + np.random.normal(0, 0.1, 3)
            cumulative_torques += base_torques + np.random.normal(0, 0.01, 3)
            
            time.sleep(0.001)  # Simulate sampling time
        
        avg_forces = cumulative_forces / num_samples
        avg_torques = cumulative_torques / num_samples
        
        self.zero_references[location] = (avg_forces, avg_torques)
        print(f"Calibration complete for {location}")
    
    def update_all_sensors(self, sensor_readings: dict):
        """Update all sensors with new readings"""
        for location, (raw_forces, raw_torques) in sensor_readings.items():
            if location in self.sensors:
                # Apply calibration
                zero_forces, zero_torques = self.zero_references[location]
                calibrated_forces = raw_forces - zero_forces
                calibrated_torques = raw_torques - zero_torques
                
                self.sensors[location].update_reading(calibrated_forces, calibrated_torques)
    
    def get_contact_status(self) -> dict:
        """Get contact status for all sensors"""
        return {loc: sensor.contact_detected for loc, sensor in self.sensors.items()}
    
    def get_support_polygon(self) -> np.ndarray:
        """Calculate support polygon from contact points"""
        contact_points = []
        
        for location, sensor in self.sensors.items():
            if sensor.contact_detected and 'foot' in location:
                # Convert sensor location to world coordinates
                # In real implementation, would use forward kinematics
                if location == 'left_foot':
                    contact_points.append([-0.1, 0.1, 0.0])  # Example position
                elif location == 'right_foot':
                    contact_points.append([0.1, -0.1, 0.0])
        
        if contact_points:
            return np.array(contact_points)
        else:
            return np.array([])

def simulate_force_torque_sensing():
    """Simulate force/torque sensor operation"""
    print("Simulating whole-body force/torque sensing...")
    
    # Initialize sensor system
    ft_system = WholeBodyForceSensor()
    
    # Calibrate sensors
    for location in ['left_foot', 'right_foot', 'left_hand', 'right_hand']:
        ft_system.calibrate_sensor(location)
    
    print("\nStarting sensor simulation...")
    
    for step in range(100):
        # Simulate sensor readings with some contact events
        sensor_readings = {}
        
        # Simulate normal walking pattern
        if step < 50:
            # Standing phase
            left_foot_forces = np.array([0, 0, -400]) + np.random.normal(0, 5, 3)
            right_foot_forces = np.array([0, 0, -400]) + np.random.normal(0, 5, 3)
        else:
            # Single support phase
            left_foot_forces = np.array([0, 0, -800]) + np.random.normal(0, 5, 3)
            right_foot_forces = np.array([0, 0, -100]) + np.random.normal(0, 2, 3)
        
        # Hand forces (simulating object holding)
        left_hand_forces = np.array([10, 5, -20]) + np.random.normal(0, 1, 3)
        right_hand_forces = np.array([5, -10, 15]) + np.random.normal(0, 1, 3)
        
        # Torques (small values)
        zero_torques = np.random.normal(0, 0.1, 3)
        
        sensor_readings['left_foot'] = (left_foot_forces, zero_torques)
        sensor_readings['right_foot'] = (right_foot_forces, zero_torques)
        sensor_readings['left_hand'] = (left_hand_forces, zero_torques)
        sensor_readings['right_hand'] = (right_hand_forces, zero_torques)
        
        # Update all sensors
        ft_system.update_all_sensors(sensor_readings)
        
        # Check contact status
        contact_status = ft_system.get_contact_status()
        
        if step % 20 == 0:
            print(f"Step {step}:")
            for location, contacted in contact_status.items():
                if contacted:
                    sensor = ft_system.sensors[location]
                    print(f"  {location}: Contact! F={sensor.forces[:2]} N")
            
            support_polygon = ft_system.get_support_polygon()
            if len(support_polygon) > 0:
                print(f"  Support polygon: {len(support_polygon)} contact points")
            print()

if __name__ == "__main__":
    simulate_force_torque_sensing()
```

## 6.3 Exteroceptive Sensors

### 6.3.1 Vision Systems

Vision systems enable humanoid robots to perceive and understand their 3D environment.

```python
#!/usr/bin/env python3
"""
Vision system interface for humanoid robotics
Demonstrates object detection, tracking, and 3D reconstruction
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
from collections import deque

@dataclass
class ImageData:
    """Image data structure"""
    timestamp: float
    image: np.ndarray
    camera_intrinsic: np.ndarray  # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # 4x1 distortion coefficients

@dataclass
class DetectedObject:
    """Detected object information"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center_3d: Optional[np.ndarray]  # [x, y, z] in world coordinates

class VisionSystem:
    """Main vision system for humanoid robot"""
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None):
        # Camera parameters
        if camera_matrix is None:
            # Default camera matrix (simulated)
            self.camera_matrix = np.array([
                [600, 0, 320],
                [0, 600, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        if dist_coeffs is None:
            self.distortion_coeffs = np.zeros(4, dtype=np.float32)
        else:
            self.distortion_coeffs = dist_coeffs
        
        # Object detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4  # Non-maximum suppression
        
        # Feature tracking
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_features = None
        
        # Object tracking
        self.tracked_objects = {}
        self.object_id_counter = 0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def process_image(self, image: np.ndarray) -> List[DetectedObject]:
        """Process single image to detect and track objects"""
        with self.lock:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Update feature tracking
            self._update_feature_tracking(gray, image)
            
            # Detect objects
            detected_objects = self._detect_objects(image)
            
            # Track objects across frames
            self._track_objects(detected_objects)
            
            # Estimate 3D positions
            for obj in detected_objects:
                obj.center_3d = self._estimate_3d_position(obj, image)
            
            return detected_objects
    
    def _detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects using classical computer vision methods"""
        # Simple color-based object detection (for demonstration)
        # In real implementation, would use deep learning models
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        detected_objects = []
        
        # Detect red objects
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        
        # Find contours
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area
                confidence = min(0.9, area / 1000)
                
                if confidence > self.confidence_threshold:
                    detected_objects.append(DetectedObject(
                        class_name="red_object",
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center_3d=None
                    ))
        
        # Detect blue objects
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area
                confidence = min(0.9, area / 1000)
                
                if confidence > self.confidence_threshold:
                    detected_objects.append(DetectedObject(
                        class_name="blue_object",
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        center_3d=None
                    ))
        
        return detected_objects
    
    def _update_feature_tracking(self, gray: np.ndarray, image: np.ndarray):
        """Update feature tracking using optical flow"""
        if self.prev_gray is None:
            # Initialize features
            self.prev_features = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            self.prev_gray = gray.copy()
            return
        
        # Calculate optical flow
        if len(self.prev_features) > 0:
            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_features, None, **self.lk_params
            )
            
            # Select good points
            good_new = new_features[status == 1]
            good_old = self.prev_features[status == 1]
            
            # Draw optical flow
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(image, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        # Update for next frame
        self.prev_gray = gray.copy()
        self.prev_features = cv2.goodFeaturesToTrack(
            gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
    
    def _track_objects(self, detected_objects: List[DetectedObject]):
        """Track objects across frames using simple tracking"""
        # Simple tracking by position matching
        for obj in detected_objects:
            # Calculate center of bounding box
            x, y, w, h = obj.bbox
            center = np.array([x + w/2, y + h/2])
            
            # Try to match with existing tracked objects
            matched = False
            for tracked_id, tracked_info in self.tracked_objects.items():
                prev_center = tracked_info['center']
                distance = np.linalg.norm(center - prev_center)
                
                if distance < 50:  # Threshold for matching
                    # Update tracked object
                    self.tracked_objects[tracked_id]['center'] = center
                    self.tracked_objects[tracked_id]['bbox'] = obj.bbox
                    self.tracked_objects[tracked_id]['confidence'] = obj.confidence
                    matched = True
                    break
            
            if not matched:
                # Create new tracked object
                obj_id = self.object_id_counter
                self.object_id_counter += 1
                self.tracked_objects[obj_id] = {
                    'center': center,
                    'bbox': obj.bbox,
                    'confidence': obj.confidence,
                    'class': obj.class_name
                }
    
    def _estimate_3d_position(self, obj: DetectedObject, image: np.ndarray) -> np.ndarray:
        """Estimate 3D position of object using camera parameters"""
        # Calculate center of bounding box
        x, y, w, h = obj.bbox
        center_2d = np.array([x + w/2, y + h/2])
        
        # In a real implementation, would use:
        # 1. Stereo vision or depth camera
        # 2. Structure from motion
        # 3. Assumptions about object size
        
        # For simulation, assume fixed distance and convert 2D to 3D
        # using camera intrinsic parameters
        z_distance = 1.0  # Fixed distance assumption
        
        # Convert 2D image coordinates to 3D world coordinates
        x_world = (center_2d[0] - self.camera_matrix[0, 2]) * z_distance / self.camera_matrix[0, 0]
        y_world = (center_2d[1] - self.camera_matrix[1, 2]) * z_distance / self.camera_matrix[1, 1]
        
        return np.array([x_world, y_world, z_distance])
    
    def get_tracked_objects_3d(self) -> dict:
        """Get all tracked objects with 3D positions"""
        with self.lock:
            result = {}
            for obj_id, obj_info in self.tracked_objects.items():
                # Estimate 3D position using the same method as detection
                x, y, w, h = obj_info['bbox']
                center_2d = np.array([x + w/2, y + h/2])
                
                z_distance = 1.0  # Fixed distance assumption
                x_world = (center_2d[0] - self.camera_matrix[0, 2]) * z_distance / self.camera_matrix[0, 0]
                y_world = (center_2d[1] - self.camera_matrix[1, 2]) * z_distance / self.camera_matrix[1, 1]
                
                result[obj_id] = {
                    'position': np.array([x_world, y_world, z_distance]),
                    'confidence': obj_info['confidence'],
                    'class': obj_info['class'],
                    'bbox': obj_info['bbox']
                }
            return result

class StereoVisionSystem:
    """Stereo vision system for depth perception"""
    def __init__(self, baseline: float = 0.12, focal_length: float = 600.0):
        self.baseline = baseline  # Distance between cameras (meters)
        self.focal_length = focal_length  # Pixel focal length
        
        # Stereo matching parameters
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def compute_depth_map(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """Compute depth map from stereo pair"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        # Depth = (baseline * focal_length) / disparity
        with np.errstate(divide='ignore'):
            depth = (self.baseline * self.focal_length) / (disparity + 1e-6)
        
        # Set invalid values to 0
        depth[disparity <= 0] = 0
        
        return depth
    
    def get_3d_point(self, u: int, v: int, depth_map: np.ndarray) -> np.ndarray:
        """Convert 2D image coordinates + depth to 3D world coordinates"""
        if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
            return np.array([0, 0, 0])
        
        z = depth_map[v, u]
        if z <= 0:
            return np.array([0, 0, 0])
        
        # Convert to 3D coordinates using camera parameters
        x = (u - 320) * z / self.focal_length  # Assuming cx = 320
        y = (v - 240) * z / self.focal_length  # Assuming cy = 240
        
        return np.array([x, y, z])

def simulate_vision_system():
    """Simulate vision system operation"""
    print("Setting up vision system...")
    vision_system = VisionSystem()
    
    print("Simulating vision processing...")
    
    for step in range(100):
        # Create a synthetic image with some objects
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some colored objects
        cv2.circle(image, (200, 200), 30, (0, 0, 255), -1)  # Red circle
        cv2.rectangle(image, (300, 150), (350, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(image, (400, 300), 20, (0, 255, 0), -1)  # Green circle
        
        # Add some noise
        noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Process image
        detected_objects = vision_system.process_image(image)
        
        if step % 20 == 0:
            print(f"Step {step}: Detected {len(detected_objects)} objects")
            
            if detected_objects:
                for i, obj in enumerate(detected_objects[:3]):  # Show first 3
                    print(f"  Object {i+1}: {obj.class_name} at {obj.center_3d}")
            
            # Show tracked objects
            tracked = vision_system.get_tracked_objects_3d()
            print(f"  Tracked objects: {len(tracked)}")
    
    print("Vision system simulation completed.")

if __name__ == "__main__":
    simulate_vision_system()
```

### 6.3.2 LIDAR Systems

LIDAR provides precise 3D mapping and obstacle detection for humanoid robots.

```python
#!/usr/bin/env python3
"""
LIDAR interface for humanoid robotics
Demonstrates 3D mapping and obstacle detection
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
from collections import deque

@dataclass
class LIDARData:
    """LIDAR data structure"""
    timestamp: float
    ranges: np.ndarray  # Distance measurements
    intensities: np.ndarray  # Intensity values
    angles: np.ndarray  # Measurement angles
    point_cloud: np.ndarray  # 3D point cloud [x, y, z]

class LIDARSystem:
    """LIDAR system interface"""
    def __init__(self, fov: float = 2 * np.pi, resolution: float = 0.01, max_range: float = 10.0):
        self.fov = fov  # Field of view in radians
        self.resolution = resolution  # Angular resolution in radians
        self.max_range = max_range  # Maximum detection range
        
        # Calculate number of beams
        self.num_beams = int(fov / resolution)
        self.angles = np.linspace(0, fov, self.num_beams, endpoint=False)
        
        # Previous scans for filtering
        self.scan_history = deque(maxlen=5)
        
        # Transformation parameters (robot to LIDAR)
        self.lidar_offset = np.array([0.0, 0.0, 1.0])  # LIDAR is 1m above ground
        
        # Obstacle detection
        self.obstacle_threshold = 0.5  # Minimum distance for obstacle
        self.min_obstacle_size = 0.3   # Minimum size to be considered obstacle
    
    def process_scan(self, ranges: np.ndarray, intensities: np.ndarray = None) -> LIDARData:
        """Process LIDAR scan data"""
        if len(ranges) != self.num_beams:
            raise ValueError(f"Expected {self.num_beams} beams, got {len(ranges)}")
        
        # Apply noise filtering to ranges
        filtered_ranges = self._filter_scan(ranges)
        
        # Convert to 3D point cloud
        point_cloud = self._ranges_to_point_cloud(filtered_ranges)
        
        # Store in history for temporal filtering
        self.scan_history.append(filtered_ranges.copy())
        
        # Perform obstacle detection
        obstacles = self._detect_obstacles(filtered_ranges)
        
        # Create result data
        lidar_data = LIDARData(
            timestamp=time.time(),
            ranges=filtered_ranges,
            intensities=intensities if intensities is not None else np.zeros_like(ranges),
            angles=self.angles,
            point_cloud=point_cloud
        )
        
        return lidar_data, obstacles
    
    def _filter_scan(self, ranges: np.ndarray) -> np.ndarray:
        """Apply basic filtering to LIDAR scan"""
        # Remove readings beyond max range
        ranges = np.where(ranges > self.max_range, self.max_range, ranges)
        
        # Simple median filtering for noise reduction
        filtered_ranges = np.zeros_like(ranges)
        
        for i in range(len(ranges)):
            # Get neighborhood
            start_idx = max(0, i - 1)
            end_idx = min(len(ranges), i + 2)
            neighborhood = ranges[start_idx:end_idx]
            
            # Apply median filter
            filtered_ranges[i] = np.median(neighborhood)
        
        return filtered_ranges
    
    def _ranges_to_point_cloud(self, ranges: np.ndarray) -> np.ndarray:
        """Convert range measurements to 3D point cloud"""
        # Convert polar to Cartesian coordinates
        x = ranges * np.cos(self.angles)
        y = ranges * np.sin(self.angles)
        z = np.zeros_like(x) + self.lidar_offset[2]  # Fixed height
        
        # Stack to create point cloud
        point_cloud = np.column_stack([x, y, z])
        
        return point_cloud
    
    def _detect_obstacles(self, ranges: np.ndarray) -> List[Tuple[int, float, float]]:
        """Detect obstacles from LIDAR data"""
        obstacles = []
        
        # Find consecutive points that are close (potential obstacles)
        distance_threshold = self.obstacle_threshold
        current_obstacle = []
        
        for i, distance in enumerate(ranges):
            if distance < distance_threshold and distance > 0.1:  # Valid range
                current_obstacle.append((i, distance))
            else:
                # End of potential obstacle
                if len(current_obstacle) >= 3:  # At least 3 consecutive points
                    # Calculate obstacle center and size
                    angles = [self.angles[obs[0]] for obs in current_obstacle]
                    avg_angle = np.mean(angles)
                    avg_distance = np.mean([obs[1] for obs in current_obstacle])
                    
                    # Estimate obstacle size
                    size = len(current_obstacle) * distance_threshold  # Rough estimate
                    
                    if size >= self.min_obstacle_size:
                        obstacles.append((int(avg_angle / self.resolution), avg_distance, size))
                
                current_obstacle = []
        
        return obstacles
    
    def create_occupancy_grid(self, lidar_data: LIDARData, resolution: float = 0.1) -> np.ndarray:
        """Create occupancy grid from LIDAR data"""
        # Determine grid size based on max range
        grid_size = int(2 * self.max_range / resolution)
        grid = np.zeros((grid_size, grid_size))
        
        # Center of grid corresponds to robot position
        grid_center = grid_size // 2
        
        # Convert point cloud to grid coordinates
        x_points = lidar_data.point_cloud[:, 0]
        y_points = lidar_data.point_cloud[:, 1]
        
        x_grid = ((x_points / resolution) + grid_center).astype(int)
        y_grid = ((y_points / resolution) + grid_center).astype(int)
        
        # Mark obstacle cells
        valid_indices = (
            (x_grid >= 0) & (x_grid < grid_size) & 
            (y_grid >= 0) & (y_grid < grid_size)
        )
        
        x_valid = x_grid[valid_indices]
        y_valid = y_grid[valid_indices]
        
        grid[x_valid, y_valid] = 1.0  # Occupied
        
        return grid
    
    def get_free_space(self, lidar_data: LIDARData) -> Tuple[float, float, float]:
        """Get available free space in front, left, right directions"""
        front_idx = slice(int(0.4 * self.num_beams), int(0.6 * self.num_beams))
        left_idx = slice(int(0.15 * self.num_beams), int(0.35 * self.num_beams))
        right_idx = slice(int(0.65 * self.num_beams), int(0.85 * self.num_beams))
        
        front_distances = lidar_data.ranges[front_idx]
        left_distances = lidar_data.ranges[left_idx]
        right_distances = lidar_data.ranges[right_idx]
        
        front_free = np.min(front_distances)
        left_free = np.min(left_distances)
        right_free = np.min(right_distances)
        
        return front_free, left_free, right_free

class HumanoidNavigationSystem:
    """Navigation system using LIDAR data"""
    def __init__(self):
        self.lidar_system = LIDARSystem()
        self.path_history = deque(maxlen=100)
        self.goal_position = None
        
    def set_goal(self, x: float, y: float):
        """Set navigation goal position"""
        self.goal_position = np.array([x, y])
    
    def process_navigation(self, lidar_data: LIDARData):
        """Process LIDAR data for navigation"""
        if self.goal_position is None:
            return "stop"  # No goal set
        
        # Get free space information
        front_free, left_free, right_free = self.lidar_system.get_free_space(lidar_data)
        
        # Calculate direction to goal
        current_pos = np.array([0, 0])  # Robot is at origin in LIDAR frame
        goal_direction = self.goal_position - current_pos
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        
        # Simple navigation logic
        if front_free < 0.8:
            # Obstacle ahead
            if left_free > right_free:
                return "turn_left"
            else:
                return "turn_right"
        elif abs(goal_angle) > 0.3:
            # Need to turn toward goal
            if goal_angle > 0:
                return "turn_left"
            else:
                return "turn_right"
        else:
            # Clear path toward goal
            return "forward"

def simulate_lidar_navigation():
    """Simulate LIDAR-based navigation"""
    print("Setting up LIDAR navigation system...")
    nav_system = HumanoidNavigationSystem()
    nav_system.set_goal(5.0, 0.0)  # Goal 5m ahead
    
    print("Simulating LIDAR navigation...")
    
    for step in range(100):
        # Simulate LIDAR scan with some obstacles
        ranges = np.full(628, 10.0)  # 628 beams for 2*pi fov at 0.01 rad resolution
        
        # Add some obstacles
        if 250 < step < 350:  # Simulate hallway with obstacle
            # Add obstacle on the right
            obstacle_start = int(471)  # ~3*pi/2 angle
            obstacle_end = int(524)    # ~5*pi/3 angle
            ranges[obstacle_start:obstacle_end] = 0.5  # 0.5m distance
        
        # Add some random noise
        ranges += np.random.normal(0, 0.02, len(ranges))
        ranges = np.clip(ranges, 0.1, 10.0)  # Valid range 0.1-10m
        
        # Process scan
        lidar_data, obstacles = nav_system.lidar_system.process_scan(ranges)
        
        # Get navigation command
        command = nav_system.process_navigation(lidar_data)
        
        if step % 20 == 0:
            front, left, right = nav_system.lidar_system.get_free_space(lidar_data)
            print(f"Step {step}: Command={command}, Free space - F:{front:.2f}, L:{left:.2f}, R:{right:.2f}")
            print(f"  Detected obstacles: {len(obstacles)}")
    
    print("LIDAR navigation simulation completed.")

def visualize_lidar_data():
    """Visualize LIDAR point cloud data"""
    lidar_system = LIDARSystem()
    
    # Create sample scan with obstacle
    ranges = np.full(628, 5.0)  # 5m in all directions
    
    # Add an obstacle
    obstacle_start = 300
    obstacle_end = 320
    ranges[obstacle_start:obstacle_end] = 1.0  # 1m distance
    
    # Process scan
    lidar_data, obstacles = lidar_system.process_scan(ranges)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot point cloud
    ax1.scatter(lidar_data.point_cloud[:, 0], lidar_data.point_cloud[:, 1], s=1, alpha=0.6)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('LIDAR Point Cloud')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Plot range scan
    ax2.plot(np.degrees(lidar_data.angles), lidar_data.ranges)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('LIDAR Range Scan')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_lidar_navigation()
    visualize_lidar_data()
```

## 6.4 Sensor Fusion and Integration

### 6.4.1 Kalman Filter for Sensor Fusion

```python
#!/usr/bin/env python3
"""
Sensor fusion using Extended Kalman Filter for humanoid robotics
Combines IMU, vision, and encoders for robust state estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time

@dataclass
class RobotState:
    """Robot state vector: [x, y, theta, vx, vy, omega]"""
    position: np.ndarray  # [x, y]
    orientation: float    # theta
    velocity: np.ndarray  # [vx, vy]
    angular_velocity: float  # omega
    covariance: np.ndarray  # State covariance matrix

class ExtendedKalmanFilter:
    """Extended Kalman Filter for robot state estimation"""
    def __init__(self):
        # State vector: [x, y, theta, vx, vy, omega]
        self.state_dim = 6
        self.state = np.zeros(self.state_dim)
        
        # Initial covariance
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.001, 0.1, 0.1, 0.01])
        
        # Measurement noise for different sensors
        self.R_imu = np.diag([0.01, 0.01, 0.001])  # [ax, ay, omega]
        self.R_vision = np.diag([0.1, 0.1, 0.01])  # [x, y, theta]
        self.R_encoders = np.diag([0.05, 0.05])     # [vx, vy]
    
    def predict(self, dt: float, control_input: np.ndarray = None):
        """Prediction step of EKF"""
        # State transition model (constant velocity model with orientation)
        theta = self.state[2]
        vx = self.state[3]
        vy = self.state[4]
        
        # Update state based on motion model
        self.state[0] += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        self.state[1] += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        self.state[2] += self.state[5] * dt  # Update orientation from angular velocity
        # Velocities remain approximately constant in simple model
        
        # Jacobian of state transition
        F = np.eye(self.state_dim)
        F[0, 2] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt
        F[0, 3] = np.cos(theta) * dt
        F[0, 4] = -np.sin(theta) * dt
        
        F[1, 2] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        F[1, 3] = np.sin(theta) * dt
        F[1, 4] = np.cos(theta) * dt
        
        F[2, 5] = dt
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_with_imu(self, measurement: np.ndarray):
        """Update with IMU measurement [ax, ay, omega]"""
        # Measurement model: extract acceleration and angular velocity from state
        # For this example, we'll use a simplified model
        H = np.zeros((3, self.state_dim))
        H[2, 5] = 1  # Measure angular velocity directly
        
        # Expected measurement
        expected = np.array([0, 0, self.state[5]])  # [ax_est, ay_est, omega]
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_with_vision(self, measurement: np.ndarray):
        """Update with vision measurement [x, y, theta] - position and orientation"""
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # Measure x position
        H[1, 1] = 1  # Measure y position  
        H[2, 2] = 1  # Measure orientation
        
        # Expected measurement
        expected = self.state[0:3]  # [x, y, theta]
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_vision
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_with_encoders(self, measurement: np.ndarray):
        """Update with encoder measurement [vx, vy] - linear velocities"""
        H = np.zeros((2, self.state_dim))
        H[0, 3] = 1  # Measure vx
        H[1, 4] = 1  # Measure vy
        
        # Expected measurement
        expected = self.state[3:5]  # [vx, vy]
        
        # Innovation
        y = measurement - expected
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_encoders
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def get_robot_state(self) -> RobotState:
        """Get current robot state"""
        return RobotState(
            position=self.state[0:2],
            orientation=self.state[2],
            velocity=self.state[3:4],
            angular_velocity=self.state[5],
            covariance=self.P
        )

class MultiSensorFusion:
    """Multi-sensor fusion system for humanoid robot"""
    def __init__(self):
        self.ekf = ExtendedKalmanFilter()
        self.last_update_time = time.time()
        
        # Sensor data buffers
        self.imu_buffer = []
        self.vision_buffer = []
        self.encoder_buffer = []
        
        # Update rates
        self.imu_rate = 100  # Hz
        self.vision_rate = 30  # Hz
        self.encoder_rate = 50  # Hz
    
    def process_imu_data(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Process IMU data and update state estimate"""
        current_time = time.time()
        
        # Combine accelerometer and gyroscope data
        measurement = np.concatenate([accel[0:2], [gyro[2]]])  # [ax, ay, omega_z]
        
        # Update EKF
        self.ekf.predict(dt)
        self.ekf.update_with_imu(measurement)
        
        self.last_update_time = current_time
    
    def process_vision_data(self, position: np.ndarray, orientation: float, dt: float):
        """Process vision data and update state estimate"""
        current_time = time.time()
        
        # Create measurement vector
        measurement = np.array([position[0], position[1], orientation])
        
        # Update EKF
        self.ekf.predict(dt)
        self.ekf.update_with_vision(measurement)
        
        self.last_update_time = current_time
    
    def process_encoder_data(self, velocity: np.ndarray, dt: float):
        """Process encoder data and update state estimate"""
        current_time = time.time()
        
        # Create measurement vector
        measurement = velocity  # [vx, vy]
        
        # Update EKF
        self.ekf.predict(dt)
        self.ekf.update_with_encoders(measurement)
        
        self.last_update_time = current_time
    
    def get_fused_state(self) -> RobotState:
        """Get the fused state estimate"""
        return self.ekf.get_robot_state()

def simulate_sensor_fusion():
    """Simulate multi-sensor fusion"""
    print("Simulating sensor fusion...")
    
    fusion_system = MultiSensorFusion()
    
    dt = 0.01  # 100Hz simulation
    
    # Simulate robot moving in a circle
    for step in range(1000):
        current_time = step * dt
        
        # Simulate robot motion (circular path)
        true_pos = np.array([2 * np.cos(0.5 * current_time), 2 * np.sin(0.5 * current_time)])
        true_vel = np.array([-np.sin(0.5 * current_time), np.cos(0.5 * current_time)]) * 1.0
        true_theta = 0.5 * current_time  # Heading angle
        true_omega = 0.5  # Angular velocity
        
        # Add noise to simulate real sensors
        imu_accel = np.array([
            -0.5 * np.sin(0.5 * current_time), 
            0.5 * np.cos(0.5 * current_time)
        ]) + np.random.normal(0, 0.01, 2)
        
        imu_gyro = np.array([0, 0, true_omega]) + np.random.normal(0, 0.001, 3)
        
        vision_pos = true_pos + np.random.normal(0, 0.05, 2)
        vision_theta = true_theta + np.random.normal(0, 0.01)
        
        encoder_vel = true_vel + np.random.normal(0, 0.02, 2)
        
        # Process sensor data
        fusion_system.process_imu_data(imu_accel, imu_gyro, dt)
        if step % 3 == 0:  # Vision at 33Hz
            fusion_system.process_vision_data(vision_pos, vision_theta, dt)
        if step % 2 == 0:  # Encoders at 50Hz
            fusion_system.process_encoder_data(encoder_vel, dt)
        
        # Get fused state
        fused_state = fusion_system.get_fused_state()
        
        if step % 100 == 0:
            pos_error = np.linalg.norm(fused_state.position - true_pos)
            vel_error = np.linalg.norm(fused_state.velocity - true_vel)
            
            print(f"Step {step}:")
            print(f"  True: pos={true_pos}, vel={true_vel}")
            print(f"  Fused: pos={fused_state.position}, vel={fused_state.velocity}")
            print(f"  Error: pos={pos_error:.3f}, vel={vel_error:.3f}")
            print()

if __name__ == "__main__":
    simulate_sensor_fusion()
```

## 6.5 ROS2 Integration for Sensor Systems

### 6.5.1 Sensor Data Publishing

```python
#!/usr/bin/env python3
"""
ROS2 sensor interface for humanoid robotics
Publishes sensor data for the perception pipeline
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, PointCloud2, PointField
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import numpy as np
import time
from collections import deque

class HumanoidSensorNode(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_node')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/laser_cloud', 10)
        
        # Timer for publishing
        self.timer = self.create_timer(0.01, self.publish_sensor_data)  # 100Hz
        
        # Joint names for humanoid robot
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
        
        # Initialize joint positions
        self.joint_positions = np.zeros(len(self.joint_names))
        self.joint_velocities = np.zeros(len(self.joint_names))
        self.joint_efforts = np.zeros(len(self.joint_names))
        
        # IMU state
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.linear_acceleration = [0.0, 0.0, 0.0]
        
        # Point cloud simulation
        self.point_cloud = self._generate_sample_pointcloud()
        
        self.get_logger().info('Humanoid Sensor Node Started')

    def _generate_sample_pointcloud(self):
        """Generate sample point cloud data"""
        # Create a simple point cloud with some objects
        points = []
        
        # Ground plane
        for i in range(10):
            for j in range(10):
                x = i * 0.5 - 2.5
                y = j * 0.5 - 2.5
                z = -1.0
                points.extend([x, y, z])
        
        # Add an obstacle
        for i in range(5):
            for j in range(5):
                x = 2.0
                y = i * 0.2 - 0.4
                z = j * 0.2
                points.extend([x, y, z])
        
        return np.array(points).astype(np.float32)

    def publish_sensor_data(self):
        """Publish all sensor data"""
        # Update joint positions with some movement
        t = self.get_clock().now().nanoseconds / 1e9
        for i in range(len(self.joint_names)):
            self.joint_positions[i] = 0.1 * np.sin(t + i * 0.1)
            self.joint_velocities[i] = 0.1 * np.cos(t + i * 0.1)
            self.joint_efforts[i] = 0.05 * np.sin(2 * t + i * 0.1)
        
        # Update IMU data
        self.orientation[3] = np.cos(t * 0.5)  # w component
        self.angular_velocity[2] = 0.5 * np.sin(t * 0.5)  # z rotation
        self.linear_acceleration[0] = 0.5 * np.cos(t)  # x acceleration
        
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = self.joint_names
        joint_msg.position = self.joint_positions.tolist()
        joint_msg.velocity = self.joint_velocities.tolist()
        joint_msg.effort = self.joint_efforts.tolist()
        
        self.joint_state_pub.publish(joint_msg)
        
        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.orientation.x = self.orientation[0]
        imu_msg.orientation.y = self.orientation[1]
        imu_msg.orientation.z = self.orientation[2]
        imu_msg.orientation.w = self.orientation[3]
        imu_msg.angular_velocity.x = self.angular_velocity[0]
        imu_msg.angular_velocity.y = self.angular_velocity[1]
        imu_msg.angular_velocity.z = self.angular_velocity[2]
        imu_msg.linear_acceleration.x = self.linear_acceleration[0]
        imu_msg.linear_acceleration.y = self.linear_acceleration[1]
        imu_msg.linear_acceleration.z = self.linear_acceleration[2]
        
        self.imu_pub.publish(imu_msg)
        
        # Publish point cloud
        pc_msg = PointCloud2()
        pc_msg.header = Header()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = 'laser_frame'
        pc_msg.height = 1
        pc_msg.width = len(self.point_cloud) // 3
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12
        pc_msg.row_step = 12 * len(self.point_cloud) // 3
        pc_msg.is_dense = True
        pc_msg.data = self.point_cloud.tobytes()
        
        self.pointcloud_pub.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_node = HumanoidSensorNode()
    
    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6.6 Laboratory Exercises

### Lab Exercise 1: IMU Calibration and Orientation Estimation

**Objective**: Calibrate IMU sensors and implement orientation estimation using sensor fusion.

**Equipment Required**:
- Inertial Measurement Unit (IMU)
- Microcontroller or single-board computer
- Computer with visualization software

**Implementation Steps**:
1. Implement accelerometer bias estimation
2. Implement gyroscope drift correction
3. Develop a complementary filter for orientation estimation
4. Test with different movement patterns
5. Analyze accuracy and drift over time

### Lab Exercise 2: Vision-Based Object Tracking

**Objective**: Implement real-time object tracking using camera vision.

**Implementation Steps**:
1. Set up camera interface with proper calibration
2. Implement color-based object detection
3. Develop tracking algorithm using optical flow
4. Test with moving objects in various lighting conditions
5. Measure tracking accuracy and latency

### Lab Exercise 3: LIDAR-based Navigation

**Objective**: Implement obstacle detection and path planning using LIDAR data.

**Implementation Steps**:
1. Interface with LIDAR sensor to acquire scan data
2. Implement point cloud processing for obstacle detection
3. Create occupancy grid mapping
4. Implement simple navigation algorithm
5. Test navigation in simulated or real environment

## 6.7 Sensor Selection and Integration Guidelines

### 6.7.1 Requirements Analysis

When selecting sensors for humanoid robots, consider:

**Performance Requirements**:
- Update rate and latency requirements
- Accuracy and precision needs
- Operating range and field of view
- Environmental conditions (temperature, humidity, lighting)

**Integration Considerations**:
- Mounting location and accessibility
- Wiring and communication requirements
- Power consumption and thermal management
- Safety and reliability requirements

### 6.7.2 Sensor Redundancy and Safety

- Critical functions should have redundant sensors
- Cross-validate sensor readings for consistency
- Implement sensor health monitoring
- Design graceful degradation when sensors fail

## 6.8 Summary

Sensor systems are fundamental to humanoid robotics, providing the information necessary for perception, navigation, and interaction. This chapter has covered:

- Proprioceptive sensors (encoders, IMU, force/torque) for state monitoring
- Exteroceptive sensors (vision, LIDAR) for environment perception  
- Sensor fusion techniques for robust state estimation
- Integration approaches using ROS2 frameworks
- Practical implementation examples with code

The quality and integration of sensors directly impacts the robot's ability to operate safely and effectively in human environments.

## 6.9 Discussion Questions

1. How does sensor redundancy improve the reliability of humanoid robots?
2. What are the key challenges in fusing data from multiple sensor types?
3. How do environmental conditions affect sensor performance in humanoid robotics?
4. What safety considerations are important when designing sensor systems?
5. How might emerging sensor technologies change humanoid robotics in the future?
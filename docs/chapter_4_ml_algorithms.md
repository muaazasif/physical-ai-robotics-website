# Chapter 4: Machine Learning and AI Algorithms for Humanoid Robotics

## 4.1 Introduction to Machine Learning in Humanoid Robotics

Machine learning (ML) and artificial intelligence (AI) are revolutionizing humanoid robotics by enabling robots to learn from experience, adapt to new situations, and perform complex tasks that are difficult to program explicitly. Unlike traditional robotics approaches that rely on pre-defined rules and mathematical models, ML algorithms allow humanoid robots to improve their performance through interaction with the environment.

The application of ML in humanoid robotics faces unique challenges:
- **High-dimensional state and action spaces**: Humanoid robots have many degrees of freedom
- **Real-time constraints**: Control decisions must be made rapidly to maintain balance
- **Safety requirements**: Learning algorithms must ensure robot and human safety
- **Continuous action domains**: Most robot actions are continuous rather than discrete
- **Sim-to-real transfer**: Models trained in simulation must work in the real world

### Types of Machine Learning in Humanoid Robotics

1. **Supervised Learning**: Learning from labeled data (e.g., pose estimation, object recognition)
2. **Reinforcement Learning**: Learning through trial and error with rewards (e.g., locomotion, manipulation)
3. **Unsupervised Learning**: Discovering patterns in unlabeled data (e.g., behavior clustering, anomaly detection)
4. **Imitation Learning**: Learning by observing and mimicking demonstrations (e.g., skill acquisition)

## 4.2 Theoretical Foundations of ML for Robotics

### 4.2.1 Markov Decision Processes (MDPs)

MDPs provide the mathematical framework for sequential decision-making in robotics:
- **State Space (S)**: All possible states the robot can be in
- **Action Space (A)**: All possible actions the robot can take
- **Transition Model (T)**: Probability of transitioning to next state given current state and action
- **Reward Function (R)**: Immediate reward for taking an action in a state
- **Discount Factor (γ)**: Weight for future rewards

For humanoid robots, states often include joint positions, velocities, sensor readings, and environment information. Actions include joint torques, desired positions, or high-level commands.

### 4.2.2 Deep Learning in Robotics

Deep learning has transformed robotics by enabling end-to-end learning of complex behaviors:

**Convolutional Neural Networks (CNNs)** for perception: Processing visual and sensory data
**Recurrent Neural Networks (RNNs)** for temporal sequences: Learning from time-series sensor data
**Deep Reinforcement Learning (DRL)** for control: Learning complex motor skills

### 4.2.3 Learning from Demonstration (LfD)

LfD allows robots to learn skills by observing human demonstrations:
- **Behavioral Cloning**: Directly mapping observations to actions
- **Inverse Optimal Control**: Learning the underlying reward function
- **Dynamics Modeling**: Learning the movement dynamics

## 4.3 Reinforcement Learning for Humanoid Control

### 4.3.1 Deep Q-Networks (DQN) and Extensions

DQN and its variants have shown success in learning humanoid behaviors:

```python
#!/usr/bin/env python3
"""
Deep Q-Network implementation for humanoid robot control
Demonstrates basic DQN training for simple locomotion task
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import gym

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class HumanoidDQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimpleHumanoidEnv:
    """Simple simulation environment for humanoid learning"""
    def __init__(self):
        self.state_size = 12  # 6 joint positions + 6 joint velocities
        self.action_size = 4  # 4 discretized actions: move forward, backward, balance, step
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Random initial joint positions and velocities
        self.state = np.random.uniform(-0.1, 0.1, self.state_size)
        self.time_step = 0
        self.max_steps = 200
        return self.state
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Simple physics simulation
        joint_positions = self.state[:6]
        joint_velocities = self.state[6:]
        
        # Apply action (simplified)
        if action == 0:  # Move forward
            joint_velocities[0] += 0.05
        elif action == 1:  # Move backward
            joint_velocities[0] -= 0.05
        elif action == 2:  # Balance
            # Try to keep joints near neutral position
            joint_positions *= 0.95
        elif action == 3:  # Step
            joint_positions[1] += 0.1  # Simple stepping motion
        
        # Update state with some physics
        new_positions = joint_positions + joint_velocities * 0.02
        new_velocities = joint_velocities * 0.98  # Damping
        
        self.state = np.concatenate([new_positions, new_velocities])
        
        # Calculate reward (encourage balanced, forward motion)
        balance_reward = 1.0 - min(abs(self.state[0]), 1.0)  # Stay balanced
        forward_reward = max(0, self.state[0]) * 0.1  # Move forward
        stability_reward = max(0, 1.0 - np.std(self.state[6:]))  # Stable movement
        
        reward = balance_reward + forward_reward + stability_reward
        
        self.time_step += 1
        done = self.time_step >= self.max_steps or abs(self.state[0]) > 1.5
        
        return self.state, reward, done, {}

def train_dqn_agent():
    """Train DQN agent on simple humanoid environment"""
    env = SimpleHumanoidEnv()
    agent = HumanoidDQNAgent(env.state_size, env.action_size)
    
    episodes = 1000
    scores = deque(maxlen=100)
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for time_step in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # Update target network every 100 episodes
        if e % 100 == 0:
            agent.update_target_network()
        
        if e % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {e}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    print("Training completed!")

if __name__ == "__main__":
    train_dqn_agent()
```

### 4.3.2 Policy Gradient Methods

Policy gradient methods optimize the policy directly, making them suitable for continuous control:

```python
#!/usr/bin/env python3
"""
Policy Gradient implementation for humanoid continuous control
Uses Actor-Critic method for stable learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Mean and std of action distribution
        self.action_mean = nn.Linear(hidden_size, action_size)
        self.action_std = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.action_mean(x))  # Actions in [-1, 1]
        std = torch.clamp(torch.exp(self.action_std(x)), 0.01, 1.0)  # Clamped std
        
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.cpu().numpy(), action_logprob.cpu().numpy()
    
    def update(self, memory):
        """Update policy using PPO algorithm"""
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Convert to tensors
        old_states = torch.FloatTensor(memory.states).to(self.device)
        old_actions = torch.FloatTensor(memory.actions).to(self.device)
        old_logprobs = torch.FloatTensor(memory.logprobs).to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            mean, std = self.actor(old_states)
            dist = Normal(mean, std)
            logprobs = dist.log_prob(old_actions)
            state_values = self.critic(old_states)
            
            # Calculate ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(state_values.squeeze(), rewards)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ContinuousHumanoidEnv:
    """Environment with continuous action space"""
    def __init__(self):
        self.state_size = 12  # 6 joint positions + 6 velocities
        self.action_size = 6  # Continuous joint control
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-0.1, 0.1, self.state_size)
        self.time_step = 0
        self.max_steps = 200
        return self.state
    
    def step(self, action):
        # Scale action to reasonable ranges
        action_scaled = np.clip(action, -1.0, 1.0) * 0.1  # Small control steps
        
        # Simple integration of joint control
        joint_positions = self.state[:6]
        joint_velocities = self.state[6:]
        
        new_velocities = joint_velocities + action_scaled * 0.02
        new_positions = joint_positions + new_velocities * 0.02
        
        self.state = np.concatenate([new_positions, new_velocities])
        
        # Reward based on balance and movement
        balance_reward = 1.0 - min(np.abs(self.state[0:3]).max(), 1.0)
        stability_reward = 1.0 - np.std(self.state[6:]) * 2.0
        
        reward = balance_reward + stability_reward * 0.5
        
        self.time_step += 1
        done = self.time_step >= self.max_steps or np.abs(self.state[:3]).max() > 1.5
        
        return self.state, reward, done, {}

def train_ppo_agent():
    """Train PPO agent on continuous humanoid environment"""
    env = ContinuousHumanoidEnv()
    agent = PPOAgent(env.state_size, env.action_size)
    memory = Memory()
    
    episodes = 1000
    max_timesteps = 200
    update_timestep = 2000  # Update policy every 2000 timesteps
    
    timestep = 0
    
    for i_episode in range(episodes):
        state = env.reset()
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Select action
            action, action_logprob = agent.select_action(state)
            
            # Take action
            state, reward, done, _ = env.step(action)
            
            # Store in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            if done:
                break
        
        # Update policy if enough timesteps have passed
        if timestep % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}, Completed")

if __name__ == "__main__":
    train_ppo_agent()
```

## 4.4 Imitation Learning and Behavior Cloning

Imitation learning enables robots to learn complex behaviors by observing demonstrations:

```python
#!/usr/bin/env python3
"""
Imitation Learning implementation for humanoid skill acquisition
Demonstrates behavior cloning from expert demonstrations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

class ImitationNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ImitationNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Add batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)  # Output continuous actions

class ImitationLearningAgent:
    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = ImitationNet(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train(self, states, actions, epochs=100, batch_size=64):
        """Train the imitation learning model"""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device)
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                predicted_actions = self.net(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def predict(self, state):
        """Predict action for given state"""
        self.net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.net(state_tensor).cpu().numpy()[0]
        return action
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.net.state_dict(), path)
    
    def load_model(self, path):
        """Load trained model"""
        self.net.load_state_dict(torch.load(path, map_location=self.device))

def generate_demonstration_data(env, num_demonstrations=1000):
    """Generate demonstration data from random expert-like actions"""
    states = []
    actions = []
    
    for _ in range(num_demonstrations):
        state = env.reset()
        action = np.random.uniform(-0.1, 0.1, env.action_size)  # Random expert action
        
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        
        if done:
            env.reset()
    
    return np.array(states), np.array(actions)

def train_imitation_model():
    """Train imitation learning model"""
    env = ContinuousHumanoidEnv()  # Using the environment from previous example
    
    # Generate demonstration data
    print("Generating demonstration data...")
    states, actions = generate_demonstration_data(env, num_demonstrations=5000)
    
    # Create and train agent
    agent = ImitationLearningAgent(env.state_size, env.action_size)
    
    print("Training imitation model...")
    agent.train(states, actions, epochs=200)
    
    # Save the trained model
    agent.save_model("imitation_model.pth")
    print("Model saved as imitation_model.pth")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_env = ContinuousHumanoidEnv()
    state = test_env.reset()
    total_reward = 0
    
    for step in range(100):
        action = agent.predict(state)
        state, reward, done, _ = test_env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Test episode reward: {total_reward}")

if __name__ == "__main__":
    train_imitation_model()
```

## 4.5 Deep Learning for Perception and Control

### 4.5.1 Vision-Based Control

Deep learning enables robots to use visual information for control:

```python
#!/usr/bin/env python3
"""
Vision-based control using CNNs for humanoid robotics
Demonstrates end-to-end learning from camera images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import random

class VisionNet(nn.Module):
    def __init__(self, action_size):
        super(VisionNet, self).__init__()
        
        # Convolutional layers for visual processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened feature map
        # Assuming input image is 84x84x3
        self.feature_size = self._get_conv_output_size((3, 84, 84))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionControlAgent:
    def __init__(self, action_size, lr=1e-4):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = VisionNet(action_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def preprocess_image(self, image):
        """Preprocess camera image for network input"""
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Add batch and channel dimensions
        return np.expand_dims(normalized, axis=[0, -1]).transpose(0, 3, 1, 2)
    
    def select_action(self, image):
        """Select action based on input image"""
        self.net.eval()
        with torch.no_grad():
            processed_image = self.preprocess_image(image)
            image_tensor = torch.FloatTensor(processed_image).to(self.device)
            action_values = self.net(image_tensor)
            action = torch.argmax(action_values, dim=1).item()
        return action

def simulate_camera_image():
    """Simulate camera image for vision-based control"""
    # Create a synthetic image with simple patterns
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Add some geometric shapes to provide visual features
    cv2.rectangle(image, (50, 50), (100, 100), (255, 0, 0), -1)  # Blue square
    cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
    cv2.line(image, (0, 200), (224, 200), (0, 0, 255), 2)  # Red line
    
    return image

def test_vision_control():
    """Test vision-based control system"""
    agent = VisionControlAgent(action_size=4)  # 4 actions: forward, backward, left, right
    
    print("Testing vision-based control...")
    
    # Simulate a few steps of vision-based control
    for step in range(10):
        # Simulate camera input
        camera_image = simulate_camera_image()
        
        # Select action based on image
        action = agent.select_action(camera_image)
        
        action_names = ["Move Forward", "Move Backward", "Turn Left", "Turn Right"]
        print(f"Step {step + 1}: Action selected - {action_names[action]}")
        
        # In a real implementation, this would control the robot
        # and receive new images as feedback

if __name__ == "__main__":
    test_vision_control()
```

## 4.6 ROS2 Integration with Machine Learning

Integrating machine learning models with ROS2 for real-time humanoid robotics:

```python
#!/usr/bin/env python3
"""
ROS2 node for machine learning inference in humanoid robotics
Integrates trained models with ROS2 for real-time control
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2

class MLInferenceNode(Node):
    def __init__(self):
        super().__init__('ml_inference_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )
        
        # Publishers
        self.action_pub = self.create_publisher(Float64MultiArray, '/robot_actions', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize ML model (example: simple policy network)
        self.policy_network = self.load_policy_model()
        
        # Robot state
        self.current_joint_positions = []
        self.current_joint_velocities = []
        self.camera_image = None
        
        # Timer for inference
        self.timer = self.create_timer(0.05, self.inference_loop)  # 20 Hz
        
        self.get_logger().info('ML Inference Node Started')

    def load_policy_model(self):
        """Load trained policy model"""
        # In a real implementation, this would load a saved model
        # For example: return torch.load('trained_policy.pth')
        
        # For this example, create a simple dummy model
        class DummyPolicy:
            def predict(self, state):
                # Dummy policy: return random action
                return np.random.uniform(-1, 1, 6).astype(np.float32)
        
        return DummyPolicy()
    
    def joint_state_callback(self, msg):
        """Update joint state from sensor feedback"""
        self.current_joint_positions = list(msg.position)
        self.current_joint_velocities = list(msg.velocity)
    
    def camera_callback(self, msg):
        """Process camera image"""
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
    
    def prepare_state_vector(self):
        """Prepare state vector for ML inference"""
        # Combine joint positions and velocities
        if (len(self.current_joint_positions) > 0 and 
            len(self.current_joint_velocities) > 0):
            
            # Example: use first 12 joint states for a 6-DOF system
            joint_data = (self.current_joint_positions[:6] + 
                         self.current_joint_velocities[:6])
            return np.array(joint_data, dtype=np.float32)
        else:
            # Return neutral pose if no joint data available
            return np.zeros(12, dtype=np.float32)
    
    def inference_loop(self):
        """Main inference loop"""
        # Prepare state for inference
        state = self.prepare_state_vector()
        
        # Get action from policy
        action = self.policy_network.predict(state)
        
        # Publish action
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)
        
        # Also publish as Twist for basic navigation
        twist_msg = Twist()
        twist_msg.linear.x = action[0] * 0.5  # Scale action to velocity
        twist_msg.linear.y = action[1] * 0.5
        twist_msg.angular.z = action[2] * 0.5
        self.cmd_vel_pub.publish(twist_msg)
        
        self.get_logger().debug(f'Inference: {action[:3]}')  # Log first 3 actions

def main(args=None):
    rclpy.init(args=args)
    ml_node = MLInferenceNode()
    
    try:
        rclpy.spin(ml_node)
    except KeyboardInterrupt:
        pass
    finally:
        ml_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4.7 Laboratory Exercises

### Lab Exercise 1: Reinforcement Learning Environment Setup

**Objective**: Set up and train a reinforcement learning agent for basic humanoid locomotion.

**Equipment Required**:
- Computer with Python and PyTorch installed
- Mujoco or PyBullet physics simulator
- ROS2 environment

**Implementation Steps**:
1. Install and configure MuJoCo or PyBullet simulator
2. Create humanoid robot model with appropriate sensors and actuators
3. Implement reward function for locomotion (forward progress, energy efficiency, balance)
4. Train DDPG or PPO agent for walking
5. Evaluate trained agent in simulation
6. Analyze learning curves and robot behavior

### Lab Exercise 2: Imitation Learning from Demonstrations

**Objective**: Implement imitation learning to teach a humanoid robot a manipulation skill.

**Implementation Steps**:
1. Record human demonstrations of a simple task (e.g., reaching, grasping)
2. Preprocess demonstration data to extract relevant features
3. Implement behavior cloning network
4. Train on demonstration dataset
5. Test learned policy on physical or simulated robot
6. Evaluate performance and compare to expert demonstrations

### Lab Exercise 3: Vision-Based Control System

**Objective**: Develop a vision-based control system for humanoid navigation.

**Implementation Steps**:
1. Set up camera system and image acquisition
2. Implement CNN-based obstacle detection
3. Create navigation policy using visual input
4. Integrate with ROS2 navigation stack
5. Test in simulated and real environments
6. Analyze robustness to different lighting conditions

## 4.8 Advanced Topics in ML for Humanoid Robotics

### 4.8.1 Multi-Task Learning

Training robots to perform multiple tasks simultaneously using shared representations.

### 4.8.2 Transfer Learning

Applying knowledge learned in one domain to similar but different tasks.

### 4.8.3 Meta-Learning

Enabling robots to learn new skills quickly from few examples.

### 4.8.4 Safe Learning

Ensuring safety during the learning process in real-world deployments.

## 4.9 Challenges and Considerations

### 4.9.1 Data Efficiency

ML algorithms often require extensive training data. For humanoid robots, this can mean thousands of hours of interaction. Techniques like domain randomization and sim-to-real transfer help address this.

### 4.9.2 Real-Time Performance

Humanoid robots require fast decision-making. Efficient neural network architectures and optimization techniques are crucial for real-time performance.

### 4.9.3 Safety and Robustness

ML-based controllers must be robust to unexpected situations and ensure safety. This includes proper handling of edge cases and failure modes.

### 4.9.4 Interpretability

Understanding and interpreting ML-based behaviors is important for debugging and trust. Techniques like attention mechanisms and explainable AI help.

## 4.10 Summary

Machine learning and AI algorithms are transforming humanoid robotics by enabling adaptive, learning-based control systems. This chapter has covered fundamental concepts including reinforcement learning, imitation learning, and deep learning applications specifically for humanoid robots. The examples demonstrate practical implementations of these concepts with code examples and integration strategies.

The future of humanoid robotics lies in the seamless integration of learning algorithms that enable robots to continuously improve their performance and adapt to new situations. As these systems become more sophisticated, they will play an increasingly important role in practical applications.

## 4.11 Discussion Questions

1. What are the main advantages of reinforcement learning over traditional control methods for humanoid robotics?
2. How can we ensure safety when deploying learning-based controllers on physical humanoid robots?
3. What are the challenges of applying transfer learning from simulation to real robots?
4. How might multi-modal learning (visual, tactile, proprioceptive) improve humanoid robot performance?
5. What role does hardware design play in enabling effective machine learning for humanoid robots?
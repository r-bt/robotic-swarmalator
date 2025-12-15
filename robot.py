from network import Node, Network
from dataclasses import dataclass
from typing import List
import numpy as np
import uuid


@dataclass
class Obstacle:
    """
    An obstacle in the environment
    """
    position: tuple  # (x, y)
    radius: float


@dataclass
class NeighbourState:
    """
    Class which tracks robot state
    """

    id: uuid.UUID
    position: tuple
    phase: float  # phase in radians


class Robot(Node):
    """
    A robot which implements the swarmalator model.
    """

    def __init__(self, network: Network, position=(0, 0), phase=0.0, natural_frequency=0.0, obstacles=None):
        """
        Initialize the robot
        """
        self._network = network

        # Robot physical properties
        self._state = NeighbourState(id=uuid.uuid4(), position=position, phase=phase)

        # Swarmalator properties
        self._K = 1.0
        self._J = 1.0
        self._A = 1.0
        self._B = 1.0
        self._natural_frequency = natural_frequency

        # Keep track of other robots
        self._neighbours: List[NeighbourState] = []
        
        # Obstacle avoidance
        self._obstacles = obstacles if obstacles is not None else []
        self._collision_threshold = 0.3  # Distance threshold for collision concern
        self._avoidance_strength = 0.8  # How strongly to avoid obstacles
        self._max_avoidance_speed = 1.5  # Limit maximum avoidance velocity
        
        # Velocity smoothing
        self._velocity = (0.0, 0.0)  # Track current velocity
        self._velocity_damping = 0.3  # Damping factor (0-1, higher = more damping)

        # Start the robot
        self._network.join(self)
        self._network.broadcast(self._state)

    @property
    def position(self):
        return self._state.position

    @property
    def phase(self):
        return self._state.phase
    
    def _calculate_collision_probability(self, obstacle: Obstacle) -> float:
        """
        Calculate collision probability based on distance to obstacle.
        Returns a value between 0 and 1, where 1 means imminent collision.
        """
        distance = np.sqrt(
            (obstacle.position[0] - self._state.position[0]) ** 2
            + (obstacle.position[1] - self._state.position[1]) ** 2
        )
        
        # Distance from robot to obstacle surface
        distance_to_surface = distance - obstacle.radius
        
        if distance_to_surface <= 0:
            return 1.0  # Already colliding
        
        if distance_to_surface > self._collision_threshold:
            return 0.0  # Too far to care
        
        # Use smooth polynomial decay instead of exponential
        # This reduces aggressive repulsion at medium distances
        normalized_dist = distance_to_surface / self._collision_threshold
        collision_prob = (1 - normalized_dist) ** 2
        return collision_prob
    
    def _calculate_steering_angle(self, obstacle: Obstacle) -> tuple:
        """
        Calculate steering vector to avoid obstacle.
        Returns a unit vector pointing away from the obstacle.
        """
        # Vector from obstacle to robot
        dx = self._state.position[0] - obstacle.position[0]
        dy = self._state.position[1] - obstacle.position[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 1e-10:
            # If on top of obstacle, pick a random direction
            angle = np.random.uniform(0, 2 * np.pi)
            return (np.cos(angle), np.sin(angle))
        
        # Normalize to get unit vector away from obstacle
        return (dx / distance, dy / distance)
    
    def _calculate_obstacle_avoidance(self) -> tuple:
        """
        Calculate the obstacle avoidance velocity component.
        Returns (v_x, v_y) weighted by collision probabilities.
        """
        total_avoidance_x = 0.0
        total_avoidance_y = 0.0
        total_weight = 0.0
        
        for obstacle in self._obstacles:
            collision_prob = self._calculate_collision_probability(obstacle)
            
            if collision_prob > 0.01:  # Only consider significant threats
                steering = self._calculate_steering_angle(obstacle)
                
                # Weight the steering by collision probability
                total_avoidance_x += steering[0] * collision_prob
                total_avoidance_y += steering[1] * collision_prob
                total_weight += collision_prob
        
        if total_weight > 0:
            # Normalize and scale by avoidance strength
            avoid_x = (total_avoidance_x / total_weight) * self._avoidance_strength
            avoid_y = (total_avoidance_y / total_weight) * self._avoidance_strength
            
            # Cap the avoidance speed to prevent excessive velocities
            avoid_speed = np.sqrt(avoid_x**2 + avoid_y**2)
            if avoid_speed > self._max_avoidance_speed:
                scale = self._max_avoidance_speed / avoid_speed
                avoid_x *= scale
                avoid_y *= scale
            
            return (avoid_x, avoid_y)
        
        return (0.0, 0.0)

    def step(self, dt):
        """
        Update the robot state for one time step
        """
        delta_phase_sum = 0
        delta_v_x_sum = 0
        delta_v_y_sum = 0

        for neighbour in self._neighbours:
            theta_diff = neighbour.phase - self._state.phase
            distance = np.sqrt(
                (neighbour.position[0] - self._state.position[0]) ** 2
                + (neighbour.position[1] - self._state.position[1]) ** 2
            )

            if distance == 0:
                distance = 1e-10  # Avoid division by zero

            delta_phase_sum += np.sin(theta_diff) / distance
            delta_v_x_sum += (
                (neighbour.position[0] - self._state.position[0]) / distance
            ) * (self._A + self._J * np.cos(theta_diff)) - (
                self._B
                * (neighbour.position[0] - self._state.position[0])
                / (distance**2)
            )
            delta_v_y_sum += (
                (neighbour.position[1] - self._state.position[1]) / distance
            ) * (self._A + self._J * np.cos(theta_diff)) - (
                self._B
                * (neighbour.position[1] - self._state.position[1])
                / (distance**2)
            )

        if len(self._neighbours) > 0:
            delta_phase_sum *= self._K / len(self._neighbours)
            delta_v_x_sum /= len(self._neighbours)
            delta_v_y_sum /= len(self._neighbours)

        # Update phase
        new_phase = (
            self._state.phase + dt * (self._natural_frequency + delta_phase_sum)
        ) % (2 * np.pi)
        
        # Calculate obstacle avoidance
        avoidance_x, avoidance_y = self._calculate_obstacle_avoidance()
        
        # Combine swarmalator velocity with obstacle avoidance
        # The avoidance term is already weighted by collision probability
        target_v_x = delta_v_x_sum + avoidance_x
        target_v_y = delta_v_y_sum + avoidance_y
        
        # Apply velocity smoothing to reduce oscillations
        # Blend previous velocity with target velocity
        smoothed_v_x = (1 - self._velocity_damping) * target_v_x + self._velocity_damping * self._velocity[0]
        smoothed_v_y = (1 - self._velocity_damping) * target_v_y + self._velocity_damping * self._velocity[1]
        
        # Store current velocity for next iteration
        self._velocity = (smoothed_v_x, smoothed_v_y)

        # Update position with smoothed velocity
        new_position = (
            self._state.position[0] + dt * smoothed_v_x,
            self._state.position[1] + dt * smoothed_v_y,
        )

        self._state = NeighbourState(
            id=self._state.id, position=new_position, phase=new_phase
        )

    def broadcast(self):
        """
        Broadcast the current state to the network
        """
        self._network.broadcast(self._state)

    def receive(self, message):
        """
        Receive a message from the network
        """
        if isinstance(message, NeighbourState):
            found = False

            if message.id == self._state.id:
                return  # Ignore self messages

            for neighbour in self._neighbours:
                if neighbour.id == message.id:
                    # Update the neighbour state
                    neighbour.position = message.position
                    neighbour.phase = message.phase
                    found = True
                    break

            if not found:
                # Add the neighbour state
                self._neighbours.append(message)
        else:
            raise ValueError("Message must be of type NeighbourState")

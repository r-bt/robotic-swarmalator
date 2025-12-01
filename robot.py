from network import Node, Network
from dataclasses import dataclass
from typing import List
import numpy as np
import uuid
from typing import Tuple
import pdb

"""
At 0.25m from the plane want to start ramping up the repulsive force

Therefore, we rescale all distances to + (1 - 0.25) so that 1/d becomes expoential at 0.25m not 1m
"""
PLANE_TURNING_POINT = 0.25 

@dataclass
class NeighbourState:
    """
    Class which tracks robot state
    """

    id: uuid.UUID
    position: np.ndarray
    phase: float  # phase in radians

@dataclass
class ExperimentalParameters:
    """
    Class which tracks experimental parameters
    """

    K: float
    J_1: float
    J_2: float
    A: List[float]
    B: List[float]
    planes: List

class Robot(Node):
    """
    A robot which implements the swarmalator model.
    """

    def __init__(self, 
        network: Network, 
        position=np.array([0, 0, 0]), 
        phase=0.0, 
        natural_frequency=0.0, 
        experimental_parameters=ExperimentalParameters(K=0.0, J_1=1.0, J_2=0.0, A=1.0, B=1.0, planes=[]),
    ):
        """
        Initialize the robot
        """
        self._network = network

        # Robot physical properties
        self._state = NeighbourState(id=uuid.uuid4(), position=position, phase=phase)

        # Swarmalator properties
        self._K = experimental_parameters.K
        self._J_1 = experimental_parameters.J_1
        self._J_2 = experimental_parameters.J_2
        self._A = experimental_parameters.A
        self._B = experimental_parameters.B
        self._natural_frequency = natural_frequency
        self._alpha = 1.0

        # Keep track of other robots
        self._neighbours: List[NeighbourState] = []

        # Keep track of planes
        self._planes = experimental_parameters.planes

        # Keep track of the current target
        self._target: np.ndarray = None

        # Start the robot
        self._network.join(self)
        self._network.broadcast(self._state)

    def _get_J1_value(self):
        """
        If target is set, calculate the J1 value based on the distance to the target
        """

        positions = np.array([neighbour.position for neighbour in self._neighbours] + [self.position])

        # Target is (3,) while positions is (agents, 3) so we need to broadcast to perform the subtraction

        distToTargetVector = self._target - positions[:, :3]

        # Calculate the distance to the target
        distToTarget = np.linalg.norm(distToTargetVector, axis=1)

        # Calculate the min and max distance to the target
        minDistToTarget = np.min(distToTarget)
        maxDistToTarget = np.max(distToTarget)

        J_val = self._alpha * (np.absolute(distToTarget[-1] - minDistToTarget)) / (maxDistToTarget - minDistToTarget)

        return J_val

    @property
    def position(self):
        return self._state.position

    @property
    def phase(self):
        return self._state.phase

    def step(self, dt):
        """
        Update the robot state for one time step
        """
        delta_phase_sum = 0
        net_force = np.zeros(3, dtype=float)

        phase = self.phase
        clean_position = self.position
        position = clean_position

        J1 = self._J_1 

        if self._target is not None:
            J1 = self._get_J1_value()

        for neighbour in self._neighbours:
            theta_diff = neighbour.phase - phase
            distance = np.sqrt(np.sum((neighbour.position - position) ** 2))

            if distance == 0:
                distance = 1e-10  # Avoid division by zero

            delta_phase_sum += np.sin(theta_diff) / distance

            attractive_force = (neighbour.position - position) / distance * (self._A + J1 * np.cos(theta_diff))
            repulsive_force = (neighbour.position - position) / distance**2 * (self._B - self._J_2 * np.cos(theta_diff))

            net_force += attractive_force - repulsive_force

        # Add plane repulsive forces

        plane_repulsive_force = np.zeros(3, dtype=float)

        for plane in self._planes:
            point = plane[0]
            normal = plane[1]

            distance = np.dot((position - point), normal) + (1 - PLANE_TURNING_POINT)

            plane_repulsive_force += (normal / (distance ** 2)) * max(*self._B) # This is simplified, but comes out to distance ** 3

        if len(self._neighbours) > 0:
            delta_phase_sum *= self._K / len(self._neighbours)
            net_force /= len(self._neighbours)

        net_force += (plane_repulsive_force / len(self._planes)) if len(self._planes) > 0 else 0

        # Add random noise
        # net_force += np.random.normal(0, 0.01, 3)

        # Update phase
        new_phase = (
            phase + dt * (self._natural_frequency + delta_phase_sum)
        ) % (2 * np.pi)

        # Update position
        new_position = position + dt * net_force

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
        
    def set_target(self, target: Tuple[float, float, float]):
        """
        Set a target position for the robot
        """
        if target is None:
            self._target = None
        else:
            self._target = np.array(target)

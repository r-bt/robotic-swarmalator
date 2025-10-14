from network import Node, Network
from dataclasses import dataclass
from typing import List
import numpy as np
import numpy.typing as npt
import uuid
from typing import Tuple


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

        # Keep track of other robots
        self._neighbours: List[NeighbourState] = []

        # Keep track of planes
        self._planes = experimental_parameters.planes

        # Start the robot
        self._network.join(self)
        self._network.broadcast(self._state)

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

        for neighbour in self._neighbours:
            theta_diff = neighbour.phase - phase
            distance = np.sqrt(np.sum((neighbour.position - position) ** 2))

            if distance == 0:
                distance = 1e-10  # Avoid division by zero

            delta_phase_sum += np.sin(theta_diff) / distance

            attractive_force = (neighbour.position - position) / distance * (self._A + self._J_1 * np.cos(theta_diff))
            repulsive_force = (neighbour.position - position) / distance**2 * (self._B - self._J_2 * np.cos(theta_diff))

            net_force += attractive_force - repulsive_force

        # Add plane repulsive forces

        for plane in self._planes:
            point = plane[0]
            normal = plane[1]

            distance = np.dot((position - point), normal)

            plane_repulsive_force = np.zeros(3, dtype=float)
            plane_repulsive_force += normal / distance * max(*self._B)

            net_force += plane_repulsive_force

        if len(self._neighbours) > 0:
            delta_phase_sum *= self._K / len(self._neighbours)
            net_force /= len(self._neighbours) + len(self._planes)

        # Add random noise
        net_force += np.random.normal(0, 0.01, 3)

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

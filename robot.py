from network import Node, Network
from dataclasses import dataclass
from typing import List
import numpy as np
import uuid


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

    def __init__(self, network: Network, position=(0, 0), phase=0.0):
        """
        Initialize the robot
        """
        self._network = network

        # Robot physical properties
        self._state = NeighbourState(id=uuid.uuid4(), position=position, phase=phase)

        # Swarmalator properties
        self._K = 0.0
        self._J = 1.0
        self._A = 1.0
        self._B = 1.0
        self._natural_frequency = 0.0

        # Keep track of other robots
        self._neighbours: List[NeighbourState] = []

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

        # Update position
        new_position = (
            self._state.position[0] + dt * delta_v_x_sum,
            self._state.position[1] + dt * delta_v_y_sum,
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

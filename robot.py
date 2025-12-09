from network import Node, Network
from dataclasses import dataclass
from typing import List
import numpy as np
import uuid
import cvxpy as cp # Required for the Quadratic Program
from typing import Tuple # For type hinting in the cbf method


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
        self._K = 1.0
        self._J = 1.0
        self._A = 1.0
        self._B = 1.0
        self._natural_frequency = 0.0

        # Keep track of other robots
        self._neighbours: List[NeighbourState] = []

        # Start the robot
        self._network.join(self)
        self._network.broadcast(self._state)

        # --- CBF Parameters ---
        self._rMin = 4  # Inner annulus radius
        self._rMax = 5 # Outer annulus radius
        self._p = 10.0    # CBF parameter (alpha in LfB + alpha*B >= 0)

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

        J_safe = self.cbf(self._J)
        # J_safe = self._J

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
            ) * (self._A + J_safe * np.cos(theta_diff)) - (
                self._B
                * (neighbour.position[0] - self._state.position[0])
                / (distance**2)
            )
            delta_v_y_sum += (
                (neighbour.position[1] - self._state.position[1]) / distance
            ) * (self._A + J_safe * np.cos(theta_diff)) - (
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
        
    def cbf(self, J_target: float) -> float:
        """
        Control Barrier Function (CBF) for a single agent (decentralized).
        Computes the safe control input J via a 1D Quadratic Program (QP).
        
        Args:
            J_target: The desired coupling strength (J_star).

        Returns:
            The safe coupling strength J.
        """
        if not self._neighbours:
            return J_target
        
        xi = np.array(self.position)
        
        # Calculate centroid of all agents (including self)
        all_positions = [np.array(n.position) for n in self._neighbours] + [xi]
        centroid = np.mean(all_positions, axis=0)

        # Initialize averaged drift and control components (2D)
        Lfb0 = np.zeros(2)
        Lgb0 = np.zeros(2)

        for neighbour in self._neighbours:
            xj = np.array(neighbour.position)
            theta_diff = neighbour.phase - self.phase
            
            pos_diff = xj - xi
            dist = np.linalg.norm(pos_diff)

            if dist < 1e-10: 
                continue

            # Calculate average drift 
            Lfb0 += self._A * pos_diff / dist - self._B * pos_diff / (dist ** 2)

            # Calculate average control
            Lgb0 += pos_diff / dist * np.cos(theta_diff)
        
        # Averaging
        N_minus_1 = len(self._neighbours)
        Lfb0 /= N_minus_1
        Lgb0 /= N_minus_1
        
        center_diff = xi - centroid # (x - x0, y - y0)
        
        # Inner circular constraint (h1 = ||x - c||^2 - rMin^2 >= 0)
        b1 = np.sum(center_diff ** 2) - self._rMin ** 2
        Lfb1 = 2 * center_diff @ Lfb0
        Lgb1 = 2 * center_diff @ Lgb0

        # Outer circular constraint (h2 = rMax^2 - ||x - c||^2 >= 0)
        b2 = self._rMax ** 2 - np.sum(center_diff ** 2)
        Lfb2 = -2 * center_diff @ Lfb0
        Lgb2 = -2 * center_diff @ Lgb0
        
        # The control input is u = J (1D variable)
        u = cp.Variable(1) 
        
        # Constraint 1 (Inner): -Lgb1 * J <= Lfb1 + p*b1
        # Constraint 2 (Outer): -Lgb2 * J <= Lfb2 + p*b2
        
        Ab = np.array([[-Lgb1], [-Lgb2]])
        b_vec = np.array([Lfb1 + self._p*b1, Lfb2 + self._p*b2])
        
        # Objective: minimize 0.5 * ||u - J_target||^2
        H = np.eye(1)
        F = -J_target
        objective = cp.Minimize(0.5*cp.quad_form(u, H) + F*u)
        constraints = [Ab @ u <= b_vec]

        prob = cp.Problem(objective, constraints)
        
        prob.solve(solver=cp.ECOS, verbose=False) 

        if u.value is None or prob.status in ["infeasible", "unbounded"]:
            return J_target
        
        return float(u.value[0])


# Set max velocity to stop agents flying off

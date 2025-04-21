from abc import ABC, abstractmethod
import random


class Node(ABC):
    """
    Abstract class for a node in the network
    """

    @abstractmethod
    def receive(self, message):
        """
        Receive a message from the network
        """
        pass


class Network:
    """
    Mimics a network of robots, a controller, and communication between them.
    """

    def __init__(self):
        """
        Initialize the network
        """
        self._nodes = []

        self._drop_probability = 0.1

    def join(self, node: Node):
        """
        Add a node to the network
        """
        self._nodes.append(node)

    def leave(self, node: Node):
        """
        Remove a node from the network
        """
        self._nodes.remove(node)

    def broadcast(self, message):
        """
        Send a message to all nodes in the network
        """
        for node in self._nodes:
            # Drop message with a probability of 0.1
            if random.random() < self._drop_probability:
                continue

            node.receive(message)

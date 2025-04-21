from network import Network
from robot import Robot
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


def angles_to_rgb(angles_rad):
    # Convert the angles to hue values (ranges from 0.0 to 1.0 in the HSV color space)
    hues = angles_rad / (2 * np.pi)

    # Set fixed values for saturation and value (you can adjust these as desired)
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Scale RGB values to 0-255 range
    rgb_colors *= 255
    rgb_colors = rgb_colors.astype(np.uint8)

    return rgb_colors


robot_count = 12


def main():
    # Create a network
    network = Network()

    # Create all the robots
    positions = np.random.uniform(-1, 1, (robot_count, 2))
    phases = np.linspace(0, 2 * np.pi, robot_count)

    robots = [Robot(network, positions[i], phases[i]) for i in range(robot_count)]

    # Setup the animation
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=20)

    dt = 0.01  # simulation time step

    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return (sc,)

    def update(frame):
        # Update all robots
        for robot in robots:
            robot.step(dt)

        for robot in robots:
            robot.broadcast()

        # Get current positions and phases
        positions = np.array([robot.position for robot in robots])
        angles = np.array([robot.phase for robot in robots])

        colors = angles_to_rgb(angles) / 255.0

        sc.set_offsets(positions)
        sc.set_color(colors)
        return (sc,)

    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=int(dt * 1000))
    plt.show()


if __name__ == "__main__":
    main()

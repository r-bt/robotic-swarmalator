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


robot_count = 9


def main():
    # Create a network
    network = Network()

    # Create all the robots
    positions = np.random.uniform(-10, 10, (robot_count, 2))

    # Along the circumference of a circle of radius 4.5
    # angles = np.linspace(0, 2 * np.pi, robot_count, endpoint=False)
    # radii = 1
    # positions = np.array([[radii * np.cos(angle), radii * np.sin(angle)] for angle in angles])

    # phases = np.linspace(0, 2 * np.pi, robot_count, endpoint=False)
    phases = np.zeros(robot_count)  # Start all phases at 0?

    robots = [Robot(network, positions[i], phases[i]) for i in range(robot_count)]

    # Setup the animation
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=20)

    ax.set_aspect('equal', adjustable='box')

    c1 = plt.Circle((0,0), 1.8, fill=False, color='black', linewidth=1, linestyle='dashed')
    c2 = plt.Circle((0,0), 2.1, fill=False, color='black', linewidth=1, linestyle='dashed')
    ax.add_patch(c1)
    ax.add_patch(c2)

    dt = 0.05  # simulation time step

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return (sc,)

    def update(frame):
        # Update all robots
        for robot in robots:
            robot.step(dt)

        for robot in robots:
            robot.broadcast()

        # Get current positions and phases
        positions = np.array([robot.position for robot in robots])

        centroid = np.mean(positions, axis=0)

        c1.set_center(centroid)
        c2.set_center(centroid)

        angles = np.array([robot.phase for robot in robots])

        colors = angles_to_rgb(angles) / 255.0

        sc.set_offsets(positions)
        sc.set_color(colors)
        return (sc,c1,c2)

    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=int(dt * 1000))
    plt.show()


if __name__ == "__main__":
    main()

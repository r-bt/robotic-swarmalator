from network import Network
from robot import Robot, Obstacle
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


robot_count = 25


def main():
    # Create a network
    network = Network()
    
    # Create obstacles
    obstacles = [
        Obstacle(position=(0.5, 0.5), radius=0.3),
        Obstacle(position=(-0.6, -0.4), radius=0.25),
        Obstacle(position=(0.8, -0.7), radius=0.2),
        Obstacle(position=(-0.5, 0.8), radius=0.15),
    ]

    # Create all the robots
    positions = np.random.uniform(-3, 3, (robot_count, 2))
    phases = np.linspace(0, 2 * np.pi, robot_count, endpoint=False)

    nat_freqs = np.ones(robot_count)
    half_len = robot_count // 2
    nat_freqs[:half_len] = -1.0  # First half slower

    robots = [Robot(network, positions[i], phases[i], nat_freqs[i], obstacles) for i in range(robot_count)]

    # Setup the animation
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=20)

    ax.set_aspect('equal', 'box')

    dt = 0.05  # simulation time step

    def init():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        # Draw obstacles
        for obstacle in obstacles:
            circle = plt.Circle(
                obstacle.position, 
                obstacle.radius, 
                color='red', 
                alpha=0.3, 
                zorder=1
            )
            ax.add_patch(circle)
        
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
        sc.set_zorder(2)  # Robots on top of obstacles
        return (sc,)

    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=int(dt * 1000))
    plt.show()


if __name__ == "__main__":
    main()

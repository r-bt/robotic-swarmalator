from network import Network
from robot import Robot, ExperimentalParameters
import numpy as np
import colorsys
from vispy import scene, app
from vispy.geometry import MeshData
import time


def angles_to_rgb(angles_rad):
    # Convert the angles to hue values (0..1 in HSV)
    hues = (angles_rad % (2 * np.pi)) / (2 * np.pi)

    # Fixed saturation and value
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Return RGB as float32 in 0..1 for VisPy
    return rgb_colors.astype(np.float32)

def create_plane(point, normal):
    # Step 2: find tangent vectors
    if abs(normal[0]) < 0.9:
        v = np.array([1, 0, 0], dtype=float)
    else:
        v = np.array([0, 1, 0], dtype=float)

    t1 = np.cross(normal, v); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1); t2 /= np.linalg.norm(t2)

    # Step 3: build quad
    s = 1.0  # half-size of plane patch
    corners = np.array([
        point + s*( t1 + t2),
        point + s*( t1 - t2),
        point + s*(-t1 - t2),
        point + s*(-t1 + t2)
    ])

    faces = np.array([[0, 1, 2], [0, 2, 3]])

    mesh = scene.visuals.Mesh(vertices=corners, faces=faces, color=(0.5, 0.7, 1, 0.5))

    return mesh

robot_count = 10

def main():
    # Create a network
    network = Network()

    # Create all the robots with 3D positions
    positions = np.random.uniform(-1, 1, (robot_count, 3))
    positions[:, 2] = np.random.uniform(0.1, 1, robot_count)  # small z variation

    # phases = np.random.uniform(0, 2 * np.pi, robot_count)
    phases = np.linspace(0, 1 * np.pi, robot_count, endpoint=False)

    planes = [
        # (np.array([0, 0, 0]), np.array([0, 0, 1])),
        # (np.array([0, 0, 1.8]), np.array([0, 0, -1])),
        # (np.array([1.5, 0, 0]), np.array([-1, 0, 0])),
        # (np.array([0, 1.5, 0]), np.array([0, -1, 0])),
        # (np.array([-1.5,0,0]), np.array([1, 0, 0])),
        # (np.array([0,-1.5,0]), np.array([0, 1, 0])),
    ]

    experimental_parameters = ExperimentalParameters(
        K=0.0, J_1=1.0, J_2=0.0, A=[1.0,1.0,1.0], B=[0.5,0.5,0.5], planes=planes)

    natural_frequencies = np.zeros(robot_count)
    # natural_frequencies[:len(natural_frequencies) // 2] = -1.0

    robots = [Robot(network, positions[i], float(phases[i]), natural_frequency=natural_frequencies[i], experimental_parameters=experimental_parameters) for i in range(robot_count)]

    target = np.array([2.0, 0.0, 0.5])
    # target=None

    for r in robots:
        r.set_target(target)

    # Setup VisPy 3D scene
    canvas = scene.SceneCanvas(keys='interactive', size=(900, 700), show=True, title='Swarmalators 3D')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    scatter = scene.visuals.Markers(parent=view.scene)

    target_marker = scene.visuals.Markers(parent=view.scene)
    if target is not None:
        target_marker.set_data(np.array([target]), face_color=(1, 0, 0, 1), size=10.0, edge_width=0.0)

    # Initial data
    current_positions = np.array([robot.position for robot in robots], dtype=np.float32)
    current_angles = np.array([robot.phase for robot in robots], dtype=np.float32)
    current_colors = angles_to_rgb(current_angles)

    scatter.set_data(current_positions, face_color=current_colors, size=5.0, edge_width=0.0)

    # Add a floor
    # for plane in planes:
    #     view.add(create_plane(plane[0], plane[1]))

    # Add a hoop in the center of the view
    theta = np.linspace(0, 2 * np.pi, 200)

    hoop_radius=0.5

    path = np.c_[np.zeros_like(theta)+1, np.cos(theta) * hoop_radius, np.sin(theta) * hoop_radius + hoop_radius]

    ring = scene.visuals.Tube(path, radius=0.05, color=(1,0.5,0.2,1), shading="smooth");
    view.add(ring)

    # Add a 3D axis helper
    axis = scene.visuals.XYZAxis(parent=view.scene)

    dt = 0.05  # simulation time step (s)

    def update(event):
        # Update all robots
        for r in robots:
            r.step(dt)

        for r in robots:
            r.broadcast()

        # Get current positions and phases
        pos = np.array([r.position for r in robots], dtype=np.float32)
        ang = np.array([r.phase for r in robots], dtype=np.float32)
        col = angles_to_rgb(ang)

        scatter.set_data(pos, face_color=col, size=5.0, edge_width=0.0)

    timer = app.Timer(interval=dt, connect=update, start=True)
    app.run()


if __name__ == "__main__":
    main()

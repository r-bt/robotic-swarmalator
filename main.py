from network import Network
from robot import Robot, ExperimentalParameters
from utils import angles_to_rgb, create_plane
import numpy as np
from vispy import scene, app
import imageio
import json

robot_count = 10
cummulative_time = 0.0

def main():
    # Create a network
    network = Network()

    # Create all the robots with 3D positions
    positions = np.random.uniform(1, 0.2, (robot_count, 3))
    positions[:, 2] = np.zeros(robot_count)  # Start all robots on the same plane (z=0)
    # positions[:, 2] = np.random.uniform(0.1, 1, robot_count)  # small z variation

    # phases = np.random.uniform(0, 2 * np.pi, robot_count)
    phases = np.linspace(0, 1 * np.pi, robot_count, endpoint=False)

    planes = [
        # (np.array([0, 0, 0]), np.array([0, 0, 1])),/
        # (np.array([0, 0, 1.8]), np.array([0, 0, -1])),
        # (np.array([1.5, 0, 0]), np.array([-1, 0, 0])),
        # (np.array([0, 1.5, 0]), np.array([0, -1, 0])),
        # (np.array([-1.5,0,0]), np.array([1, 0, 0])),
        # (np.array([0,-1.5,0]), np.array([0, 1, 0])),
    ]

    experimental_parameters = ExperimentalParameters(
        K=0.0, J_1=1.0, J_2=0.0, A=[1.0,1.0,1.0], B=[0.6,0.6,0.6], planes=planes)

    natural_frequencies = np.zeros(robot_count)
    # natural_frequencies = np.ones(robot_count)
    # natural_frequencies[:len(natural_frequencies) // 2] = -1.0

    robots = [Robot(network, positions[i], float(phases[i]), natural_frequency=natural_frequencies[i], experimental_parameters=experimental_parameters) for i in range(robot_count)]

    target = np.array([0, 0, 0])
    # target=None

    # for r in robots:
    #     r.set_target(target)

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
    for plane in planes:
        view.add(create_plane(plane[0], plane[1]))

    # Add a hoop in the center of the view
    # theta = np.linspace(0, 2 * np.pi, 200)

    # hoop_radius=0.5

    # path = np.c_[np.zeros_like(theta)+1, np.cos(theta) * hoop_radius, np.sin(theta) * hoop_radius + hoop_radius]

    # ring = scene.visuals.Tube(path, radius=0.05, color=(1,0.5,0.2,1), shading="smooth");
    # view.add(ring)

    # Add a 3D axis helper
    axis = scene.visuals.XYZAxis(parent=view.scene)

    dt = 0.05  # simulation time step (s)

    results_file = open("results/results.jsonl", "w")
    parameters = {
        'robot_count': robot_count,
        'dt': dt,
        'K': experimental_parameters.K,
        'J_1': experimental_parameters.J_1,
        'J_2': experimental_parameters.J_2,
        'A': list(experimental_parameters.A),
        'B': list(experimental_parameters.B),
        'natural_frequencies': natural_frequencies.tolist(),
        'initial_positions': positions.tolist(),
        'initial_phases': phases.tolist(),
        'planes': [{'point': p.tolist(), 'normal': n.tolist()} for p, n in planes],
    }
    results_file.write(json.dumps(parameters) + '\n')

    results = []

    def update(event):
        global cummulative_time
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
        writer.append_data(canvas.render())

        # Record the current state (positions, phases, and timestep)
        data = {
            'time': cummulative_time,
            'positions': pos.tolist(),
            'phases': ang.tolist()
        }

        # results.append(data)
        results_file.write(json.dumps(data) + '\n')
        cummulative_time += dt

        if cummulative_time >= 30.0 and robots[0]._target is None:  # Add a target after 30 seconds
            print("Setting target at time 30s")
            for r in robots:
                r.set_target(target)

        # if cummulative_time >= 120.0:  # Run for 120 seconds
        #     app.quit()

    frame_interval = 0.01  # seconds

    writer = imageio.get_writer("results/swarmalator_simulation.mp4", fps=int(1 / frame_interval), codec="libx264", quality=8)

    timer = app.Timer(interval=frame_interval, connect=update, start=True)
    app.run()

    writer.close()
    results_file.close()
    print("Video saved to results/swarmalator_simulation.mp4")
    print("Results saved to results/results.jsonl")


if __name__ == "__main__":
    main()

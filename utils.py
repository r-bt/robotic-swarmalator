import numpy as np
import colorsys
from vispy import scene


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

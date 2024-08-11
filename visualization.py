import matplotlib.pyplot as plt
import svgwrite
import cairosvg
from scipy.interpolate import splprep, splev

def fit_bezier_curve(points, num_points=100):
    if len(points) < 2:
        return points
    tck, u = splprep([points[:, 0], points[:, 1]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine)).T

def plot_curves(paths_XYs, colours):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            if len(XY) > 2:
                bezier_curve = fit_bezier_curve(XY)
                ax.plot(bezier_curve[:, 0], bezier_curve[:, 1], c=c, linewidth=2)
                ax.fill(bezier_curve[:, 0], bezier_curve[:, 1], c=c, alpha=0.3)
            else:
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
                ax.fill(XY[:, 0], XY[:, 1], c=c, alpha=0.3)
    ax.set_aspect('equal')
    plt.show()

def plot_shapes(rectangle, circle_points, inner_points, reflected_points=None, symmetry_type="Original"):
    fig, ax = plt.subplots(figsize=(10, 8))

    if rectangle.size > 0:
        x_rect, y_rect = zip(*rectangle)
        ax.plot(x_rect, y_rect, '-', color='purple', linewidth=2, label='Outer Rectangle')

    if circle_points.size > 0:
        ax.plot(circle_points[:, 0], circle_points[:, 1], color='purple', linewidth=2, label='Connecting Circle')

    if inner_points.size > 0:
        ax.scatter(inner_points[:, 0], inner_points[:, 1], color='purple', label='Inner Points Within Circle')

    if reflected_points is not None:
        ax.scatter(reflected_points[:, 0], reflected_points[:, 1], color='orange', label=f'Reflected Points ({symmetry_type})')

    ax.set_aspect('equal')
    plt.title(f"Shapes and Connecting Circle ({symmetry_type})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def polylines2svg(paths_XYs, svg_path, colours):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W = max(W, np.max(XY[:, 0]))
            H = max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        parent_width=W,
        parent_height=H,
        output_width=fact * W,
        output_height=fact * H,
        background_color='white'
    )

def plot_curves_within_circle(curves, center, radius):
    for curve in curves:
        x_curve, y_curve = zip(*curve)
        # Check if the curve points are within the circle
        distances = np.sqrt((np.array(x_curve) - center[0])**2 + (np.array(y_curve) - center[1])**2)
        if np.all(distances <= radius):
            plt.plot(x_curve, y_curve, '--', color='green', label='Inner Bezier Curve')
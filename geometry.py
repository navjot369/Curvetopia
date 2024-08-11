import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.optimize import minimize

def fill_occlusions(paths_XYs):
    filled_paths = []
    for path in paths_XYs:
        filled_path = []
        for shape in path:
            if len(shape) > 2:
                hull = ConvexHull(shape)
                filled_shape = shape[hull.vertices]
            else:
                filled_shape = shape
            filled_path.append(filled_shape)
        filled_paths.append(filled_path)
    return filled_paths

def analyze_connectivity(paths_XYs):
    polygons = []
    for path in paths_XYs:
        for shape in path:
            if len(shape) > 2:
                polygon = Polygon(shape)
                if polygon.is_valid:
                    polygons.append(polygon)
                else:
                    repaired_polygon = make_valid(polygon)
                    polygons.append(repaired_polygon)

    if not polygons:
        return "Disconnected"

    try:
        multi_polygon = MultiPolygon(polygons)
        unified_polygon = unary_union(multi_polygon)

        if isinstance(unified_polygon, (Polygon, MultiPolygon)):
            return "Connected" if len(unified_polygon.geoms) == 1 else "Disconnected"
        elif isinstance(unified_polygon, GeometryCollection):
            valid_polygons = [geom for geom in unified_polygon.geoms if isinstance(geom, Polygon) and geom.is_valid]
            if valid_polygons:
                multi_polygon = MultiPolygon(valid_polygons)
                return "Connected" if len(multi_polygon.geoms) == 1 else "Disconnected"
        return "Disconnected"
    except Exception as e:
        print(f"Error in connectivity analysis: {e}")
        return "Disconnected"

def separate_shapes(fragments):
    all_points = np.concatenate(fragments)

    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    margin = 5

    outer_points = []
    inner_points = []

    for point in all_points:
        if (point[0] <= min_x + margin or point[0] >= max_x - margin or
            point[1] <= min_y + margin or point[1] >= max_y - margin):
            outer_points.append(point)
        else:
            inner_points.append(point)

    return np.array(outer_points), np.array(inner_points)

def fit_rectangle(points):
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]

def fit_circle(points):
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    theta = np.linspace(0, 2 * np.pi, 100)
    return [(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)) for t in theta], center, radius

def filter_points_within_circle(points, center, radius):
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
    return points[distances < radius]

def fit_circle_to_ellipse_points(points, center, radius):
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
    boundary_points = points[np.abs(distances - radius) < 10]

    if len(boundary_points) < 3:
        print("Not enough boundary points to fit a circle.")
        return None, None

    def circle_fit_error(params):
        c_x, c_y, r = params
        return np.sum((np.sqrt((boundary_points[:, 0] - c_x)**2 +
                               (boundary_points[:, 1] - c_y)**2) - r)**2)

    initial_guess = [center[0], center[1], radius]
    result = minimize(circle_fit_error, initial_guess, method='Nelder-Mead')
    new_center_x, new_center_y, new_radius = result.x

    theta = np.linspace(0, 2 * np.pi, 100)
    new_circle_points = np.column_stack([
        new_center_x + new_radius * np.cos(theta),
        new_center_y + new_radius * np.sin(theta)
    ])

    return new_circle_points, (new_center_x, new_center_y, new_radius)

def reflect_points_across_vertical(points, x_line):
    return np.array([[2 * x_line - x, y] for x, y in points])

def reflect_points_across_horizontal(points, y_line):
    return np.array([[x, 2 * y_line - y] for x, y in points])

def reflect_points_across_line(points, line_point, line_direction):
    line_point = np.array(line_point)
    line_direction = np.array(line_direction) / np.linalg.norm(line_direction)
    reflected_points = []
    for point in points:
        point = np.array(point)
        projection_length = np.dot(point - line_point, line_direction)
        projection = line_point + projection_length * line_direction
        reflection = 2 * projection - point
        reflected_points.append(reflection)
    return np.array(reflected_points)

def group_fragments(fragments):
    coords = np.concatenate(fragments, axis=0)  # Combine all fragments into one array
    clustering = DBSCAN(eps=5, min_samples=2).fit(coords)
    labels = clustering.labels_

    grouped_fragments = {}
    for label, coord in zip(labels, coords):
        if label not in grouped_fragments:
            grouped_fragments[label] = []
        grouped_fragments[label].append(coord)
    return grouped_fragments

def merge_rectangle(fragments):
    x_coords, y_coords = zip(*fragments)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)
    ]

def merge_circle(fragments):
    x_coords, y_coords = zip(*fragments)
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    radius = np.mean([np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in fragments])
    theta = np.linspace(0, 2 * np.pi, 100)
    return [(center_x + radius * np.cos(t), center_y + radius * np.sin(t)) for t in theta]

def identify_shape(fragments):
    # Implement a simple heuristic or method to distinguish between rectangles and circles
    # This is a placeholder; you may need a more complex approach
    if len(fragments) > 20:  # Arbitrary threshold for simplicity
        return "circle"
    return "rectangle"
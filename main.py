from utils import read_csv, rdp_simplify
from geometry import (
    fill_occlusions, analyze_connectivity, separate_shapes,
    fit_rectangle, fit_circle, filter_points_within_circle,
    fit_circle_to_ellipse_points, reflect_points_across_vertical,
    reflect_points_across_horizontal, reflect_points_across_line
)
from visualization import plot_curves, plot_shapes, polylines2svg
from sklearn.cluster import DBSCAN

def process_csv_and_fill_occlusions(input_csv, colours):
    input_paths_XYs = read_csv(input_csv)
    simplified_paths_XYs = rdp_simplify(input_paths_XYs, epsilon=1.0)
    
    print("Plotting input curves...")
    plot_curves(simplified_paths_XYs, colours)
    input_result = analyze_connectivity(simplified_paths_XYs)

    filled_paths_XYs = fill_occlusions(simplified_paths_XYs)
    print("Plotting filled curves...")
    plot_curves(filled_paths_XYs, colours)
    output_result = analyze_connectivity(filled_paths_XYs)

    print(f"Input image: {input_result}")
    print(f"Output image: {output_result}")

    if input_result == "Disconnected" and output_result == "Connected":
        print("Occlusion has been filled")
    else:
        print("No change in connectivity")

    return simplified_paths_XYs

def process_fragments(input_csv):
    all_fragments = [fragment for shape_frag in read_csv(input_csv) for fragment in shape_frag]
    outer_points, inner_points = separate_shapes(all_fragments)

    rectangle = fit_rectangle(outer_points)
    circle, center, radius = fit_circle(inner_points)

    inner_points_within_circle = filter_points_within_circle(inner_points, center, radius)

    db = DBSCAN(eps=5, min_samples=10).fit(inner_points_within_circle)
    labels = db.labels_

    new_circle_points, new_circle_params = fit_circle_to_ellipse_points(inner_points_within_circle, center, radius)

    if new_circle_params:
        new_center_x, new_center_y, new_radius = new_circle_params
        distances_to_new_circle = np.sqrt((inner_points_within_circle[:, 0] - new_center_x)**2 +
                                          (inner_points_within_circle[:, 1] - new_center_y)**2)
        filtered_inner_points = inner_points_within_circle[np.abs(distances_to_new_circle - new_radius) >= 5]

    return np.array(rectangle), np.array(new_circle_points), filtered_inner_points


def find_symmetry_and_reflect(path_XYs, symmetry_type):
    reflected_paths = []
    for path in path_XYs:
        reflected_path = []
        for segment in path:
            if symmetry_type == "vertical":
                x_line = (segment[:, 0].min() + segment[:, 0].max()) / 2
                reflected_segment = reflect_points_across_vertical(segment, x_line)
            elif symmetry_type == "horizontal":
                y_line = (segment[:, 1].min() + segment[:, 1].max()) / 2
                reflected_segment = reflect_points_across_horizontal(segment, y_line)
            elif symmetry_type == "diagonal":
                line_point = [(segment[:, 0].min() + segment[:, 0].max()) / 2,
                              (segment[:, 1].min() + segment[:, 1].max()) / 2]
                line_direction = [1, 1]  # 45-degree line
                reflected_segment = reflect_points_across_line(segment, line_point, line_direction)
            reflected_path.append(reflected_segment)
        reflected_paths.append(reflected_path)
    return reflected_paths

def plot_paths_with_symmetry(original_paths, reflected_paths, colours, symmetry_type):
    fig, ax = plt.subplots(figsize=(8, 8))
    color_idx = 0
    for original_path, reflected_path in zip(original_paths, reflected_paths):
        for segment, reflected_segment in zip(original_path, reflected_path):
            if len(segment) > 0:
                segment = np.array(segment)
                ax.plot(segment[:, 0], segment[:, 1], c=colours[color_idx % len(colours)], linewidth=2)
                ax.plot(reflected_segment[:, 0], reflected_segment[:, 1], c=colours[(color_idx + 1) % len(colours)], linestyle='--', linewidth=2)
        color_idx += 1
    ax.set_aspect('equal')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid(True, which='both', color='lightgray', linestyle='--')
    plt.title(f"Original and {symmetry_type.capitalize()} Reflected Paths")
    plt.show()

def analyze_symmetry(input_csv):
    path_XYs = read_csv(input_csv)
    simplified_paths = rdp_simplify(path_XYs, epsilon=1.0)
    
    colours = ['red', 'green', 'blue']
    
    # Analyze vertical symmetry
    reflected_paths_vertical = find_symmetry_and_reflect(simplified_paths, "vertical")
    plot_paths_with_symmetry(simplified_paths, reflected_paths_vertical, colours, "vertical")

    # Analyze horizontal symmetry
    reflected_paths_horizontal = find_symmetry_and_reflect(simplified_paths, "horizontal")
    plot_paths_with_symmetry(simplified_paths, reflected_paths_horizontal, colours, "horizontal")

    # Analyze diagonal symmetry
    reflected_paths_diagonal = find_symmetry_and_reflect(simplified_paths, "diagonal")
    plot_paths_with_symmetry(simplified_paths, reflected_paths_diagonal, colours, "diagonal")

if __name__ == "__main__":
    input_csv = 'path/to/your/input.csv'
    colours = ['red', 'green']
    
    # Process CSV and fill occlusions
    simplified_paths_XYs = process_csv_and_fill_occlusions(input_csv, colours)
    polylines2svg(simplified_paths_XYs, 'output.svg', colours)
    
    # Process fragments
    rectangle, circle_points, filtered_inner_points = process_fragments(input_csv)
    plot_shapes(rectangle, circle_points, filtered_inner_points)
    
    # Analyze symmetry
    analyze_symmetry(input_csv)

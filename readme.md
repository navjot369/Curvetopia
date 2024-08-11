# Shape Processing and Symmetry Analysis

## Project Overview

This project, developed for [Hackathon Name], is a Python-based tool for processing, analyzing, and visualizing geometric shapes. It offers functionalities such as shape detection, occlusion filling, symmetry analysis, and SVG generation.

## Features

- **CSV Input Processing**: Read and parse shape data from CSV files.
- **Shape Simplification**: Implement the Ramer-Douglas-Peucker algorithm for curve simplification.
- **Occlusion Detection and Filling**: Identify and fill occluded areas in shapes.
- **Fragment Analysis**: Process and group shape fragments.
- **Symmetry Detection**: Analyze vertical, horizontal, and diagonal symmetry in shapes.
- **Visualization**: Generate plots and SVG outputs for processed shapes.
- **Bezier Curve Fitting**: Smooth shape outlines using Bezier curves.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/navjot369/Curvetopia.git
   cd shape-processing
   ```
2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

## Project Structure

```
shape_processing/
├── __init__.py
├── utils.py
├── geometry.py
├── visualization.py
└── main.py
README.md
```

## Usage

1. Prepare your input CSV file with shape data.
2. Run the main script:

   ```
   python main.py
   ```
3. The script will process the input, generate visualizations, and save output files (SVG, PNG) in the project directory.

## Modules

### utils.py

Contains utility functions for reading CSV files and applying the RDP algorithm.

### geometry.py

Implements geometric operations like shape filling, connectivity analysis, and symmetry calculations.

### visualization.py

Handles all plotting and SVG generation functionalities.

### main.py

The entry point of the application, orchestrating the entire shape processing pipeline.

## Example Output


[Include some example images or descriptions of the output here]

## Detailed Algorithm Analysis

### 1. Ramer-Douglas-Peucker (RDP) Algorithm

**Purpose**: Curve simplification
**Complexity**: O(n log n) average case, O(n^2) worst case
**Implementation**:

```python
def rdp_simplify(path_XYs, epsilon=1.0):
    simplified_paths = []
    for path in path_XYs:
        simplified_path = []
        for shape in path:
            if len(shape) > 2:
                simplified_shape = rdp(shape, epsilon=epsilon)
                simplified_path.append(simplified_shape)
            else:
                simplified_path.append(shape)
        simplified_paths.append(simplified_path)
    return simplified_paths
```

**Key Points**:

- Reduces the number of points in a curve while retaining its shape
- Epsilon parameter controls the level of simplification
- Crucial for optimizing processing speed and reducing noise in input data

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Purpose**: Clustering shape fragments
**Complexity**: O(n log n) with spatial indexing, O(n^2) without
**Implementation**:

```python
from sklearn.cluster import DBSCAN

def group_fragments(fragments):
    coords = np.concatenate(fragments, axis=0)
    clustering = DBSCAN(eps=5, min_samples=2).fit(coords)
    labels = clustering.labels_

    grouped_fragments = {}
    for label, coord in zip(labels, coords):
        if label not in grouped_fragments:
            grouped_fragments[label] = []
        grouped_fragments[label].append(coord)
    return grouped_fragments
```

**Key Points**:

- Efficiently groups nearby points without requiring a predefined number of clusters
- Robust to noise and outliers
- Parameters `eps` and `min_samples` control cluster density and size

### 3. Convex Hull Algorithm

**Purpose**: Shape filling and occlusion detection
**Complexity**: O(n log n)
**Implementation**:

```python
from scipy.spatial import ConvexHull

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
```

**Key Points**:

- Used to find the smallest convex polygon containing all points in a set
- Efficient for filling occluded areas in shapes
- Implemented using QuickHull algorithm in SciPy

### 4. Bezier Curve Fitting

**Purpose**: Smooth shape representation
**Complexity**: O(n) for evaluation, O(n^3) for least squares fitting
**Implementation**:

```python
from scipy.interpolate import splprep, splev

def fit_bezier_curve(points, num_points=100):
    if len(points) < 2:
        return points
    tck, u = splprep([points[:, 0], points[:, 1]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine)).T
```

**Key Points**:

- Creates smooth curves passing through or near given points
- Uses B-spline representation for efficient computation
- Allows for precise control over curve smoothness

### 5. Symmetry Detection Algorithm

**Purpose**: Identify vertical, horizontal, and diagonal symmetry
**Complexity**: O(n) for each symmetry type
**Implementation**:

```python
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
```

**Key Points**:

- Detects symmetry by reflecting points and comparing with original
- Implements three types of symmetry: vertical, horizontal, and diagonal
- Efficiency depends on the number of points in the shape

### 6. Circle Fitting Algorithm

**Purpose**: Fit a circle to a set of points
**Complexity**: O(n) for initial estimation, O(iterations * n) for optimization
**Implementation**:

```python
from scipy.optimize import minimize

def fit_circle_to_ellipse_points(points, center, radius):
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
    boundary_points = points[np.abs(distances - radius) < 10]

    def circle_fit_error(params):
        c_x, c_y, r = params
        return np.sum((np.sqrt((boundary_points[:, 0] - c_x)**2 +
                               (boundary_points[:, 1] - c_y)**2) - r)**2)

    initial_guess = [center[0], center[1], radius]
    result = minimize(circle_fit_error, initial_guess, method='Nelder-Mead')
    new_center_x, new_center_y, new_radius = result.x

    return new_circle_points, (new_center_x, new_center_y, new_radius)
```

**Key Points**:

- Uses optimization to find the best-fitting circle
- Minimizes the sum of squared distances from points to the circle
- Employs the Nelder-Mead method for optimization, balancing accuracy and speed

## Performance Considerations

1. **RDP Algorithm**: Critical for reducing data points. Adjust epsilon for balance between accuracy and speed.
2. **DBSCAN**: Sensitive to `eps` and `min_samples`. May require tuning for optimal performance.
3. **Convex Hull**: Efficient but may oversimplify complex shapes with concavities.
4. **Bezier Curve Fitting**: Trade-off between smoothness and computational cost. Adjust `num_points` as needed.
5. **Symmetry Detection**: Linear time complexity but may be computationally intensive for very large datasets.
6. **Circle Fitting**: Optimization-based approach may be slow for large point sets. Consider sampling or alternative methods for initial estimation.

## Algorithmic Challenges and Solutions

1. **Occlusion Handling**:

   - Challenge: Detecting and filling occluded areas in complex shapes.
   - Solution: Combination of Convex Hull for simple cases and more sophisticated region-growing algorithms for complex occlusions.
2. **Shape Classification**:

   - Challenge: Accurately distinguishing between different shape types (e.g., circles vs. rectangles).
   - Solution: Implemented a heuristic approach based on point count and distribution. Future work includes machine learning-based classification.
3. **Symmetry Detection Accuracy**:

   - Challenge: Ensuring robust symmetry detection in the presence of noise or slight asymmetries.
   - Solution: Implemented a tolerance threshold in symmetry comparisons. Considering statistical methods for more robust detection.
4. **Scalability**:

   - Challenge: Maintaining performance with large datasets.
   - Solution: Employed RDP for data reduction and optimized algorithms. Future work includes parallel processing for large-scale analyses.

## Challenges Faced

1. **Occlusion Detection**: Developing an algorithm to accurately detect and fill occluded areas in complex shapes.
2. **Performance Optimization**: Balancing processing speed with accuracy, especially for large datasets.
3. **Symmetry Analysis**: Implementing robust methods to detect various types of symmetry in irregular shapes.

## Future Improvements

1. Implement machine learning models for more accurate shape classification.
2. Add support for 3D shape processing and analysis.
3. Develop a user-friendly GUI for easier interaction with the tool.
4. Optimize algorithms for better performance with large datasets.

## Team Members

- Shresth Verma
- Animesh Chaudhri
- Navjot Singh

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# shape_processing/
# ├── __init__.py
# ├── utils.py
# ├── geometry.py
# ├── visualization.py
# └── main.py

import numpy as np
from rdp import rdp

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

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


# main.py

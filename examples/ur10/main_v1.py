import h5py
from numpy.typing import NDArray
import cv2
import numpy as np


def jpg2img(buf: bytes) -> NDArray[np.uint8]:
    """
    Decode img RGB or GRAY from jpg buffer.
    Return HWC image.
    """
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    if len(img.shape) == 2:
        # hw -> hwc for grayscale
        img = img[:,:, None]
    if img.shape[2] == 3:
        # BGR -> RGB
        img = img[:, :, [2, 1, 0]]
    return img


def read_hdf5_datasets(file_path, dataset_names, index):
    """
    Reads a specific index from multiple datasets in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_names (list of str): List of dataset names to read.
        index (int): The index to retrieve from each dataset.

    Returns:
        dict: A dictionary mapping dataset names to their values at the given index.
    """
    results = {}

    with h5py.File(file_path, "r") as f:
        for dataset_name in dataset_names:
            if dataset_name in f:
                dataset = f[dataset_name]
                if index < dataset.shape[0]:  # Ensure index is within bounds
                    results[dataset_name] = dataset[index]
                else:
                    results[dataset_name] = f"Index {index} out of bounds (shape={dataset.shape})"
            else:
                results[dataset_name] = "Dataset not found"

    return results

# Example Usage
file_path = '/app/data/dataset/01a/5bae4bdce89efa9927e12d5ddd49b5be3571f.h5py'
dataset_list = [
    "episode/observations/CompressedRGB__rgb",
    "episode/observations/array__joint_angles",
    "episode/observations/array__gripper"
]
index = 5  # Change this to the index you want to retrieve

data = read_hdf5_datasets(file_path, dataset_list, index)
data["episode/observations/CompressedRGB__rgb"] = jpg2img(data["episode/observations/CompressedRGB__rgb"])

# Print results
for name, value in data.items():
    print(f"{name}: {value.shape}")

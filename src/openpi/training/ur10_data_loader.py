import h5py
import os
import numpy as np
import cv2
import random
from numpy.typing import NDArray
from typing import List, Dict, Iterator, Tuple
import torch


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


def find_h5py_files(directory: str) -> List[str]:
    """Find all .h5py files in the given directory and its subdirectories."""
    files = []
    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if file.endswith(".h5py"):
                files.append(os.path.join(root, file))
    return files


def scan_files(files: List[str], field_name: str) -> Tuple[int, float]:
    """Scan all .h5py files to compute the number of episodes and transitions."""
    total_transitions = []
    for file in files:
        with h5py.File(file, "r") as f:
            if field_name in f:
                num_transitions = f[field_name].shape[0]
                total_transitions.append(num_transitions)

    return (
        int(np.sum(total_transitions)) if total_transitions else 0,
        np.mean(total_transitions) if total_transitions else 0,
    )


class HDF5UR10Dataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            files: List[str],
            field_list: List[str],
            num_forward_records: List[int],
            ):
        self.field_list = field_list
        self.files = files
        self.num_forward_records = num_forward_records

        # This will be set per worker by get_worker_info()
        self.rank = None
        self.world_size = None

    def print_stats(self, files: List[str]):
        num_tr, avg_tr = scan_files(files, self.field_list[0])

        """Print statistics about the dataset."""
        print(
            f"Dataset stats. Rank: {self.rank}\n"
            f"    Number of Files: {len(files)}\n"
            f"    Transitions: {num_tr:.2f}\n"
            f"    Average Transitions per File: {avg_tr:.2f}"
        )

    def _read_transition(self, file_path: str, index: int) -> Dict[str, NDArray]:
        """Read a specific transition from a given HDF5 file."""
        results = {}

        with h5py.File(file_path, "r") as f:
            for dataset_name, num_fwd_rec in zip(self.field_list, self.num_forward_records):
                if dataset_name in f:
                    dataset = f[dataset_name]
                    if index < dataset.shape[0]:
                        data = dataset[index]
                        num_records = dataset.shape[0]
                        indices = [min(index + i, num_records - 1) for i in range(num_fwd_rec)]
                        data_list = [dataset[idx] for idx in indices]
                        if "CompressedRGB" in dataset_name:
                            data_list = [jpg2img(data) for data in data_list]
                        results[dataset_name] = np.stack(data_list)
                    else:
                        results[dataset_name] = f"Index {index} out of bounds (shape={dataset.shape})"
                else:
                    results[dataset_name] = "Dataset not found"

        results["prompt"] = "pick any object in a bin"
        return results

    def __iter__(self) -> Iterator[Dict[str, NDArray]]:
        """Randomly sample files and transitions to yield."""

        # Get worker info (for sharding)
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Set the rank and world_size based on worker info
            self.rank = worker_info.id
            self.world_size = worker_info.num_workers
        else:
            # In case of single worker or no workers
            self.rank = 0
            self.world_size = 1

        # Determine files to load based on worker shard
        shard_files = self.files[self.rank::self.world_size]
        self.print_stats(shard_files)

        # Iterate over the shard's files
        while True:
            file = random.choice(shard_files)  # Randomly select a file from the worker's shard
            with h5py.File(file, "r") as f:
                num_transitions = f[self.field_list[0]].shape[0]
                index = random.randint(0, num_transitions - 1)  # Randomly select an index
                yield self._read_transition(file, index)

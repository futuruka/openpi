import typing
from openpi.training.ur10_data_loader import HDF5UR10Dataset, find_h5py_files


def main():

    # Example Usage
    directory = "/app/data/dataset/"
    field_list = [
        "episode/observations/CompressedRGB__rgb",
        "episode/observations/array__joint_angles",
        "episode/observations/array__gripper"
    ]

    dataset = HDF5UR10Dataset(
        files=find_h5py_files(directory),
        field_list=field_list
    )

    # Example of how you might use DataLoader to load data in parallel
    from torch.utils.data import DataLoader
    import torch

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2
    )

    for ind, transition in enumerate(dataloader):
        for name, value in transition.items():
            if hasattr(value, 'shape'):
                print(f"{ind} {name}: {value.shape}")
            else:
                print(f"{ind} {name}: {value}")

        if ind == 5:
            break


if __name__ == "__main__":
    main()

from openpi.training.ur10_data_loader import HDF5UR10Dataset, find_h5py_files
import cv2


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
        field_list=field_list,
        num_forward_records=[1, 51, 51],
    )

    # Example of how you might use DataLoader to load data in parallel
    from torch.utils.data import DataLoader

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

            if name == "episode/observations/CompressedRGB__rgb":
                cv2.imwrite(f'img_{ind}_0.jpg', value[0, 0].numpy()[:, :, ::-1])
                value = value.float()
                print(f'--- img min {value[:, 0].min()} mean {value[:, 0].mean()} max {value[:, 0].max()}')
            elif name == "episode/observations/array__gripper":
                print(f'--- {name}\n{value.squeeze()}')

        if ind == 5:
            break


if __name__ == "__main__":
    main()

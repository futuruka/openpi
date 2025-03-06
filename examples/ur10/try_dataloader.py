from openpi.training.ur10_data_loader import HDF5UR10Dataset, find_h5py_files
import cv2
import numpy as np


def main():

    # Example Usage
    directory = "/app/data/dataset-valid/"
    # field_list = [
    #     "episode/observations/CompressedRGB__rgb",
    #     "episode/observations/array__joint_angles",
    #     "episode/observations/array__gripper",
    #     "episode/actions/scalar__gripper|pos",
    # ]

    field_list = [
        "episode/observations/CompressedRGB__rgb",
        "episode/observations/array__gripper",
        "episode/observations/array__external_force",
        "episode/observations/array__external_torque",

        "episode/actions/array__move|rotvec",
        "episode/actions/array__move|xyz",
        "episode/actions/scalar__gripper|pos",
        "episode/actions/scalar__gripper|force",
        "episode/actions/scalar__gripper|speed",
    ]

    field_list_optional = [
        "episode/observations/array__external_force",
        "episode/observations/array__external_torque",
        "episode/actions/scalar__gripper|force",
        "episode/actions/scalar__gripper|speed",
    ]

    default_values = {
        "episode/observations/array__external_force": np.zeros((1, 3)),
        "episode/observations/array__external_torque": np.zeros((1, 3)),
        "episode/actions/scalar__gripper|force": np.array([5.] * 50),
        "episode/actions/scalar__gripper|speed": np.array([20.] * 50),
    }

    dataset = HDF5UR10Dataset(
        files=find_h5py_files(directory),
        field_list=field_list,
        field_list_optional=field_list_optional,
        default_values=default_values,
        num_forward_records=[1, 1, 1, 1, 50, 50, 50, 50, 50],
    )

    # Example of how you might use DataLoader to load data in parallel
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
    )

    for ind, transition in enumerate(dataloader):
        for name, value in transition.items():
            if hasattr(value, 'shape'):
                print(f"{ind} {name}: {value.shape}")
            else:
                print(f"{ind} {name}: {value}")

            # if name == "episode/observations/CompressedRGB__rgb":
            #     cv2.imwrite(f'img_{ind}_0.jpg', value[0, 0].numpy()[:, :, ::-1])
            #     value = value.float()
            #     print(f'--- img min {value[:, 0].min()} mean {value[:, 0].mean()} max {value[:, 0].max()}')
            # elif name == "episode/observations/array__gripper":
            #     print(f'--- {name}\n{value.squeeze()}')

        if ind == 5:
            break


if __name__ == "__main__":
    main()


# HDF5 "/extra_disk_1/parilo/data/dataset/dataset_sft_iter_2_1786/000/3468d30c7abc4e734a18f7c6133e15a10038a.h5py" {
# FILE_CONTENTS {
#  group      /
#  group      /episode
#  group      /episode/actions
#  dataset    /episode/actions/array__move|rotvec
#  dataset    /episode/actions/array__move|xyz
#  dataset    /episode/actions/array__pose
#  dataset    /episode/actions/scalar__episode_end
#  dataset    /episode/actions/scalar__gripper|force
#  dataset    /episode/actions/scalar__gripper|pos
#  dataset    /episode/actions/scalar__gripper|speed
#  dataset    /episode/actions/scalar__robot
#  dataset    /episode/dones
#  group      /episode/infos
#  group      /episode/observations
#  dataset    /episode/observations/CompressedDepth__depth
#  dataset    /episode/observations/CompressedDepth__depth_head
#  dataset    /episode/observations/CompressedDepth__depth_side
#  dataset    /episode/observations/CompressedRGB__rgb
#  dataset    /episode/observations/CompressedRGB__rgb_head
#  dataset    /episode/observations/CompressedRGB__rgb_side
#  dataset    /episode/observations/array__external_force
#  dataset    /episode/observations/array__external_torque
#  dataset    /episode/observations/array__gripper
#  dataset    /episode/observations/array__gripper_open
#  dataset    /episode/observations/array__joint_angles
#  dataset    /episode/observations/array__robot_pos_rotvec
#  dataset    /episode/rewards
#  }
# }
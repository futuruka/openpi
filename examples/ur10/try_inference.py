from openpi.training.ur10_data_loader import HDF5UR10Dataset, find_h5py_files, read_episode_data
import time
from typing import Any, Dict

import cv2
import logging
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import image_tools


IMG_SIZE = (224, 224)


# h5dump -n /app/data/dataset-valid/02b/209755180c8dc3edfd9076fcae548cbedeea2.h5py


def make_observation_ur10(
        cam_img: np.ndarray,
        joint_angles: np.ndarray,
        gripper_pos: float,
        ) -> Dict[str, Any]:
    return {
        # "observation/wrist_image": cv2.resize(cam_img[:, :, :3].astype(np.uint8), IMG_SIZE),
        "wrist_image": image_tools.resize_with_pad(cam_img[:, :, :3].astype(np.uint8), 224, 224),
        "joint_angles": joint_angles,
        "gripper_pos": np.array((gripper_pos,)),
        "prompt": "grasp object",
    }


def main():

    np.set_printoptions(suppress=True, precision=4)

    # Example Usage
    # directory = "/app/data/dataset_sft_iter_1_1688/"
    field_list = [
        "episode/observations/CompressedRGB__rgb",
        "episode/observations/array__joint_angles",
        # "episode/observations/array__gripper"
        "episode/actions/scalar__gripper|pos",
    ]

    ep = read_episode_data(
        # train
        file_path='/app/data/dataset/dataset_sft_iter_1_1688/000/9fedb8cf74024a6b3f779227961bf7c96da55.h5py',
        # valid
        # file_path='/app/data/dataset-valid/02b/209755180c8dc3edfd9076fcae548cbedeea2.h5py',
        field_list=field_list,
    )

    ep_images = ep["episode/observations/CompressedRGB__rgb"]
    ep_joint_angles = ep["episode/observations/array__joint_angles"]
    # ep_gripper_pos = ep["episode/observations/array__gripper"]
    ep_gripper_pos = ep["episode/actions/scalar__gripper|pos"]

    print(f'--- keys {list(ep.keys())}')
    print(f'--- images {ep_images.shape}')
    print(f'--- joint_angles {ep_joint_angles.shape}')
    print(f'--- gripper_pos {ep_gripper_pos.shape}')

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    num_steps = len(ep_joint_angles)
    start = time.time()
    for ind in range(num_steps):

        joint_angles = ep_joint_angles[ind]
        obs_gripper_pos = ep_gripper_pos[ind]
        # obs_gripper_pos = 99
        print(f'--- joint_angles {joint_angles} gr {obs_gripper_pos}')

        obs = make_observation_ur10(
            cam_img=ep_images[ind],
            joint_angles=joint_angles,
            gripper_pos=obs_gripper_pos,
        )

        # print(f'--- obs {joint_angles.shape} gr {ep_gripper_pos[ind].shape}')

        cv2.imwrite(f'wrist_{ind}.jpg', obs['wrist_image'][:, :, ::-1])

        ret = policy.infer(obs)
        actions = ret["actions"]
        action_gripper = actions[0, 6]
        print(f'{ind} gr {action_gripper} gr diff {obs_gripper_pos / 100 - action_gripper}')
        # print(f'{ind} gripper {actions[:, 6]}')
        # act = actions[0][:6]
        # dj = act - joint_angles
        # dj = act
        # print(f'---  a {act}')
        # print(f'--- dj {dj}')
        # dj_scale = 0.02 / np.fabs(dj).max()
        # dj_s = dj_scale * dj
        # print(f'--- dj2 {dj_s}')
        # # print(f'--- policy ret {actions.shape}\n{actions[:2]}')
        # target_j = (joint_angles + dj_s).tolist()
        # print(f'--- target_j {target_j}')

        # gripper_pos = actions[0][6]
        # gripper_pos = np.clip(gripper_pos, 0, 1)
        # print(f'--- gripper_pos {gripper_pos}')

    end = time.time()

    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / num_steps:.2f} ms")


if __name__ == "__main__":
    main()

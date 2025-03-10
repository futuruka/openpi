import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# def make_ur10_example() -> dict:
#     """Creates a random input example for the Droid policy."""
#     return {
#         # "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/joint_position": np.random.rand(6),
#         "observation/gripper_position": np.random.rand(1),
#         "prompt": "pick an item",
#     }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        print(f'--- WARNING: scale float image {image.shape} {image.dtype}')
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

                        # "wrist_image": "episode/observations/CompressedRGB__rgb",
                        # "gripper_pos": "episode/observations/array__gripper",
                        # "external_force": "episode/observations/array__external_force",
                        # "external_torque": "episode/observations/array__external_torque",

                        # "move_rotvec": "episode/actions/array__move|rotvec",
                        # "move_xyz": "episode/actions/array__move|xyz",
                        # "action_gripper_pos": "episode/actions/scalar__gripper|pos",
                        # "action_gripper_force": "episode/actions/scalar__gripper|force",
                        # "action_gripper_speed": "episode/actions/scalar__gripper|speed",

                        # "prompt": "prompt",
                        # "features": "features",

@dataclasses.dataclass(frozen=True)
class UR10Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        np.set_printoptions(suppress=True, precision=4)

        # print(f'--- joint_angles {data["joint_angles"].shape} wrist_image {data["wrist_image"].shape} gr {data["gripper_pos"].shape}')
        is_batch = len(data["gripper_pos"].shape) == 2
        # print(f'--- gripper_pos {data["gripper_pos"].shape} external_force {data["external_force"].shape} external_torque {data["external_torque"].shape}', flush=True)

        if is_batch:
            state = np.concatenate([
                data["gripper_pos"],
                data["external_force"],
                # np.zeros_like(data["external_torque"]),
            ], axis=-1)
            state = np.squeeze(state, axis=0)
        else:
            state = np.concatenate([
                data["gripper_pos"],
                data["external_force"],
                # np.zeros_like(data["external_torque"]),
            ], axis=-1)

        # print(f'--- state {state.shape} {state}', flush=True)
        # print(f'--- state {state.shape}', flush=True)
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # base_image = _parse_image(data["observation/exterior_image_1_left"])
        if is_batch:
            wrist_image = _parse_image(np.squeeze(data["wrist_image"], axis=0))
        else:
            wrist_image = _parse_image(data["wrist_image"])
        # print(f'--- img {wrist_image.shape} {wrist_image.dtype} min {wrist_image.min()} mean {wrist_image.mean()} max {wrist_image.max()}', flush=True)
        # print(f'--- state {state[:7]}')

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (np.zeros_like(wrist_image), wrist_image, np.zeros_like(wrist_image))
                image_masks = (np.False_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (np.zeros_like(wrist_image), np.zeros_like(wrist_image), wrist_image)
                image_masks = (np.False_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "move_rotvec" in data:
            actions = np.concatenate([
                data["move_rotvec"] / 100,
                data["move_xyz"] / 100,
                np.expand_dims(data["action_gripper_pos"] / 100, -1),
                np.expand_dims(data["action_gripper_force"] / 100, -1),
                np.expand_dims(data["action_gripper_speed"] / 100, -1),
            ], axis=-1)
            # print(f'--- actions {actions.shape}\n{actions[:2]}')
            # print(f'--- actions {actions.shape}', flush=True)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["features"] + data["prompt"]
        # print(f'--- features {data["features"]}', flush=True)
        # print(f'--- prompt {inputs["prompt"]}', flush=True)

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :9])}

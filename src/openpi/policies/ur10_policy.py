import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur10_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        # "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "pick an item",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        print(f'--- WARNING: scale float image {image.shape} {image.dtype}')
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR10Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # print(f'--- joint_angles {data["joint_angles"].shape} wrist_image {data["wrist_image"].shape}')
        is_batch = len(data["joint_angles"].shape) == 2

        if is_batch:
            state = np.concatenate([data["joint_angles"][0], data["gripper_pos"][0] / 100])
        else:
            state = np.concatenate([data["joint_angles"], data["gripper_pos"] / 100])

        state = transforms.pad_to_dim(state, self.action_dim)
        # print(f'--- state {state}')

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # base_image = _parse_image(data["observation/exterior_image_1_left"])
        if is_batch:
            wrist_image = _parse_image(np.squeeze(data["wrist_image"], axis=0))
        else:
            wrist_image = _parse_image(data["wrist_image"])
        # print(f'--- img {wrist_image.shape} {wrist_image.dtype} min {wrist_image.min()} mean {wrist_image.mean()} max {wrist_image.max()}')

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (np.zeros_like(wrist_image), wrist_image, np.zeros_like(wrist_image))
                image_masks = (np.False_, np.True_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # if "actions" in data:
        if is_batch:
            actions = np.concatenate([
                data["joint_angles"][1:],
                data["gripper_pos"][1:] / 100
            ], axis=-1)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions
            # print(f'--- actions {inputs["actions"].shape}')

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        print(f'--- load 1', flush=True)
        path = download.maybe_download(self.params_path)
        print(f'--- load 2', flush=True)
        loaded_params = _model.restore_params(path, restore_type=np.ndarray)
        print(f'--- load 3', flush=True)
        # Add all missing LoRA weights.
        res = _merge_params(loaded_params, params, missing_regex=".*lora.*")
        print(f'--- load 4', flush=True)
        return res


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    print(f'--- _merge_params 1', flush=True)
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    print(f'--- _merge_params 2', flush=True)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    print(f'--- _merge_params 3', flush=True)

    # First, take all weights that are a subset of the reference weights.
    print(f'--- _merge_params 4', flush=True)
    result = {}
    for k, v in flat_loaded.items():
        print(f'--- _merge_params 4.1 {k}', flush=True)
        if k in flat_ref:
            print(f'--- _merge_params 4.2 {k} v.dtype {v.dtype} flat_ref[k].dtype {flat_ref[k].dtype}', flush=True)
            if v.dtype != flat_ref[k].dtype:
                result[k] = v.astype(flat_ref[k].dtype)
            else:
                result[k] = v
        print(f'--- _merge_params 4.3', flush=True)
    print(f'--- _merge_params 5', flush=True)

    # Then, merge any missing weights as defined by the missing regex.
    print(f'--- _merge_params 6', flush=True)
    pattern = re.compile(missing_regex)
    print(f'--- _merge_params 7', flush=True)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]
    print(f'--- _merge_params 8', flush=True)

    res = flax.traverse_util.unflatten_dict(result, sep="/")
    print(f'--- _merge_params 9', flush=True)
    return res

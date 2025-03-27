import logging
import os
import time
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from openpi.training.ur10_data_loader import read_episode_data
from openpi.shared import nnx_utils


CONFIG = "pi0_ur10_finetune_n"
CHECKPOINT = "/openpi_assets/openpi-assets/checkpoints/pi0_base/"
NUM_STEPS = 10
IMG_SIZE = (224, 224)


def main():
    logging.basicConfig(level=logging.INFO, force=True)

    field_list = [
        "episode/observations/CompressedRGB__rgb",
        "episode/observations/array__joint_angles",
        "episode/observations/array__gripper",
    ]

    ep = read_episode_data(
        # train
        file_path='/app/data/dataset/dataset_sft_iter_1_1688/000/9fedb8cf74024a6b3f779227961bf7c96da55.h5py',
        # valid
        # file_path='/app/data/dataset-valid/02b/209755180c8dc3edfd9076fcae548cbedeea2.h5py',
        field_list=field_list,
    )

    # model.load_from_checkpoint("s3://openpi-assets/checkpoints/pi0_base/params")
    policy = _policy_config.create_trained_policy(
        train_config=_config.get_config(CONFIG),
        checkpoint_dir=CHECKPOINT,
    )
    # print(policy.model.PaliGemma.img)
    print(f'--- num_classes {policy.model.PaliGemma.img.module.num_classes}')

    siglip_jit =nnx_utils.module_jit(policy.model.PaliGemma.img.__call__)

    os.makedirs("siglip_cmp_data", exist_ok=True)

    for ind in range(0, 5 * NUM_STEPS, 5):
        image = ep["episode/observations/CompressedRGB__rgb"][ind]
        image = cv2.resize(image[:, :, :3].astype(np.uint8), IMG_SIZE).astype(np.float32) / 255.0
        image = image * 2.0 - 1.0
        # image = np.expand_dims(image, axis=0)
        print(f'--- {ind} image {image.shape} dtype {image.dtype} min {image.min()} max {image.max()} mean {image.mean()}')

        # image = np.random.randn(1, 224, 224, 3).astype(np.float32)
        t1 = time.time()
        image_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], image)
        tokens, out_dict = siglip_jit(image_jax)
        tokens = jax.tree.map(lambda x: np.asarray(x), tokens)
        out_dict = jax.tree.map(lambda x: np.asarray(x), out_dict)
        t2 = time.time()
        print(f'--- dt {t2 - t1}')
        print(f'--- {ind} tokens {tokens.shape} min {tokens.min()} max {tokens.max()} mean {tokens.mean()}')
        print(f'--- {ind} out_dict {out_dict.keys()}')
        print(f'--- {ind} encoded {out_dict["encoded"].shape} min {out_dict["encoded"].min()} max {out_dict["encoded"].max()} mean {out_dict["encoded"].mean()}')
        print(f'--- {ind} pre_logits {out_dict["pre_logits"].shape} min {out_dict["pre_logits"].min()} max {out_dict["pre_logits"].max()} mean {out_dict["pre_logits"].mean()}')
        print(f'--- {ind} pre_logits_2d {out_dict["pre_logits_2d"].shape} min {out_dict["pre_logits_2d"].min()} max {out_dict["pre_logits_2d"].max()} mean {out_dict["pre_logits_2d"].mean()}')

        np.save(f'siglip_cmp_data/out_dict_{ind}.npy', out_dict)
        np.save(f'siglip_cmp_data/tokens_{ind}.npy', tokens)
        np.save(f'siglip_cmp_data/image_{ind}.npy', image)


if __name__ == "__main__":
    main()

# /app# CUDA_VISIBLE_DEVICES=1 DATASET_PATH="" python examples/ur10/try_model.py
# INFO:root:Loading model...
# INFO:2025-03-27 17:06:10,223:jax._src.xla_bridge:924: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
# INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
# INFO:2025-03-27 17:06:10,224:jax._src.xla_bridge:924: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
# INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
# INFO:absl:orbax-checkpoint version: 0.11.6
# INFO:absl:Created BasePyTreeCheckpointHandler: pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=None
# INFO:absl:Restoring checkpoint from /openpi_assets/openpi-assets/checkpoints/pi0_base/params.
# INFO:absl:[thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
# INFO:absl:[process=0] /jax/checkpoint/read/bytes_per_sec: 916.8 MiB/s (total bytes: 6.0 GiB) (time elapsed: 6 seconds) (per-host)
# INFO:absl:Finished restoring checkpoint in 6.74 seconds from /openpi_assets/openpi-assets/checkpoints/pi0_base/params.
# --- cache_dir /openpi_assets
# INFO:root:Loaded norm stats from /app/assets/pi0_ur10_finetune_n/ur10
# INFO:root:Loaded norm stats from /openpi_assets/openpi-assets/checkpoints/pi0_base/assets/ur10
# --- num_classes 2048
# --- 0 image (224, 224, 3) dtype float32 min -1.0 max 0.9686274528503418 mean -0.32078424096107483
# --- dt 3.2401185035705566
# --- 0 tokens (1, 256, 2048) min -21 max 21.625 mean 0.00186157
# --- 0 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 0 encoded (1, 256, 1152) min -34.5 max 34 mean -0.000839233
# --- 0 pre_logits (1, 256, 1152) min -34.5 max 34 mean -0.000839233
# --- 0 pre_logits_2d (1, 16, 16, 1152) min -34.5 max 34 mean -0.000839233
# --- 5 image (224, 224, 3) dtype float32 min -1.0 max 0.9529411792755127 mean -0.265169233083725
# --- dt 0.07046318054199219
# --- 5 tokens (1, 256, 2048) min -21 max 21.375 mean 0.00188446
# --- 5 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 5 encoded (1, 256, 1152) min -34 max 37 mean -0.000801086
# --- 5 pre_logits (1, 256, 1152) min -34 max 37 mean -0.000801086
# --- 5 pre_logits_2d (1, 16, 16, 1152) min -34 max 37 mean -0.000801086
# --- 10 image (224, 224, 3) dtype float32 min -1.0 max 0.37254905700683594 mean -0.26306653022766113
# --- dt 0.07815146446228027
# --- 10 tokens (1, 256, 2048) min -21 max 22.75 mean 0.00185394
# --- 10 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 10 encoded (1, 256, 1152) min -34.25 max 33.5 mean -0.000968933
# --- 10 pre_logits (1, 256, 1152) min -34.25 max 33.5 mean -0.000968933
# --- 10 pre_logits_2d (1, 16, 16, 1152) min -34.25 max 33.5 mean -0.000968933
# --- 15 image (224, 224, 3) dtype float32 min -1.0 max 0.7176470756530762 mean -0.27395668625831604
# --- dt 0.07325601577758789
# --- 15 tokens (1, 256, 2048) min -22.25 max 22.75 mean 0.00179291
# --- 15 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 15 encoded (1, 256, 1152) min -33.5 max 32.75 mean -0.000858307
# --- 15 pre_logits (1, 256, 1152) min -33.5 max 32.75 mean -0.000858307
# --- 15 pre_logits_2d (1, 16, 16, 1152) min -33.5 max 32.75 mean -0.000858307
# --- 20 image (224, 224, 3) dtype float32 min -1.0 max 0.9686274528503418 mean -0.2934582233428955
# --- dt 0.07767796516418457
# --- 20 tokens (1, 256, 2048) min -22.125 max 22.5 mean 0.00179291
# --- 20 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 20 encoded (1, 256, 1152) min -33 max 30.5 mean 0.00534058
# --- 20 pre_logits (1, 256, 1152) min -33 max 30.5 mean 0.00534058
# --- 20 pre_logits_2d (1, 16, 16, 1152) min -33 max 30.5 mean 0.00534058
# --- 25 image (224, 224, 3) dtype float32 min -1.0 max 0.9921568632125854 mean -0.2672925591468811
# --- dt 0.0705881118774414
# --- 25 tokens (1, 256, 2048) min -21.75 max 22.125 mean 0.00189209
# --- 25 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 25 encoded (1, 256, 1152) min -33.5 max 28.375 mean -0.000843048
# --- 25 pre_logits (1, 256, 1152) min -33.5 max 28.375 mean -0.000843048
# --- 25 pre_logits_2d (1, 16, 16, 1152) min -33.5 max 28.375 mean -0.000843048
# --- 30 image (224, 224, 3) dtype float32 min -1.0 max 1.0 mean -0.21549950540065765
# --- dt 0.07875180244445801
# --- 30 tokens (1, 256, 2048) min -21.625 max 22 mean 0.00189209
# --- 30 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 30 encoded (1, 256, 1152) min -33.5 max 30.625 mean 0.00695801
# --- 30 pre_logits (1, 256, 1152) min -33.5 max 30.625 mean 0.00695801
# --- 30 pre_logits_2d (1, 16, 16, 1152) min -33.5 max 30.625 mean 0.00695801
# --- 35 image (224, 224, 3) dtype float32 min -1.0 max 0.9921568632125854 mean -0.213873028755188
# --- dt 0.0769648551940918
# --- 35 tokens (1, 256, 2048) min -21.5 max 22.5 mean 0.00188446
# --- 35 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 35 encoded (1, 256, 1152) min -32.75 max 31 mean 0.00695801
# --- 35 pre_logits (1, 256, 1152) min -32.75 max 31 mean 0.00695801
# --- 35 pre_logits_2d (1, 16, 16, 1152) min -32.75 max 31 mean 0.00695801
# --- 40 image (224, 224, 3) dtype float32 min -1.0 max 0.9921568632125854 mean -0.21726247668266296
# --- dt 0.07590651512145996
# --- 40 tokens (1, 256, 2048) min -20.625 max 22.125 mean 0.0018692
# --- 40 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 40 encoded (1, 256, 1152) min -32.5 max 30.625 mean 0.00695801
# --- 40 pre_logits (1, 256, 1152) min -32.5 max 30.625 mean 0.00695801
# --- 40 pre_logits_2d (1, 16, 16, 1152) min -32.5 max 30.625 mean 0.00695801
# --- 45 image (224, 224, 3) dtype float32 min -1.0 max 0.9607843160629272 mean -0.22598043084144592
# --- dt 0.07866668701171875
# --- 45 tokens (1, 256, 2048) min -20.875 max 22.375 mean 0.00189209
# --- 45 out_dict dict_keys(['encoded', 'encoder', 'logits', 'logits_2d', 'pre_logits', 'pre_logits_2d', 'stem', 'with_posemb'])
# --- 45 encoded (1, 256, 1152) min -32 max 35 mean 0.00695801
# --- 45 pre_logits (1, 256, 1152) min -32 max 35 mean 0.00695801
# --- 45 pre_logits_2d (1, 16, 16, 1152) min -32 max 35 mean 0.00695801
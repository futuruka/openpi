import jax.numpy as jnp
import numpy as np
import time


def test_jnp_np():
    print(f'test_jnp_np')
    t1 = time.time()
    asd = jnp.zeros((100, 200, 300))
    np.array(asd).shape
    print(time.time() - t1)


test_jnp_np()
test_jnp_np()
test_jnp_np()

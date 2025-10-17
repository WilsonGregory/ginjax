import itertools as it

import jax
import jax.numpy as jnp
import jax.random as random

import ginjax.geometric as geom

D = 2
print(f"D = {D}!")
key = random.PRNGKey(0)

key, subkey = random.split(key)
A = random.normal(subkey, shape=(D, D))

kd = geom.KroneckerDeltaSymbol.get(D, 2)
lc = geom.LeviCivitaSymbol.get(D)

print(A)

trace_map = jnp.einsum("ij,kl", kd, kd) / D
trace_tensor = jnp.einsum("ijkl,kl->ij", trace_map, A)
print(trace_tensor)

antisym_map = jnp.einsum("ij,kl", lc, lc) / 2
antisym_tensor = jnp.einsum("ijkl,kl->ij", antisym_map, A)
print(antisym_tensor)

sym_tensor_actual = 0.5 * (A + A.T) - (jnp.trace(A) / D) * jnp.eye(D)

sym_map = (
    0.5 * jnp.einsum("ik,jl->ijkl", kd, kd) + 0.5 * jnp.einsum("il,jk->ijkl", kd, kd) - trace_map
)
sym_tensor = jnp.einsum("ijkl,kl->ij", sym_map, A)
print(sym_tensor)
assert jnp.allclose(sym_tensor_actual, sym_tensor)

assert jnp.allclose(jnp.einsum("ijkl,ijkl", trace_map, antisym_map), 0.0)
assert jnp.allclose(jnp.einsum("ijkl,ijkl", trace_map, sym_map), 0.0)
assert jnp.allclose(jnp.einsum("ijkl,ijkl", antisym_map, sym_map), 0.0)

D = 3
print(f"D = {D}!")
key, subkey = random.split(key)
A = random.normal(subkey, shape=(D, D))

kd = geom.KroneckerDeltaSymbol.get(D, 2)
lc = geom.LeviCivitaSymbol.get(D)

print(A)

trace_map = jnp.einsum("ij,kl", kd, kd) / D
trace_tensor = jnp.einsum("ijkl,kl->ij", trace_map, A)
print(trace_tensor)

antisym_psvec_map = lc / 2
antisym_pseudovector = jnp.einsum("ijk,jk->i", antisym_psvec_map, A)
print(antisym_pseudovector)

antisym_map = 0.5 * jnp.einsum("ik,jl->ijkl", kd, kd) - 0.5 * jnp.einsum("il,jk->ijkl", kd, kd)
antisym_tensor = jnp.einsum("ijkl,kl->ij", antisym_map, A)
print(antisym_tensor)

sym_tensor_actual = 0.5 * (A + A.T) - (jnp.trace(A) / D) * jnp.eye(D)

# 5 independent components, but you can put it onto a matrix so that it transforms like a matrix
sym_map = (
    0.5 * jnp.einsum("ik,jl->ijkl", kd, kd) + 0.5 * jnp.einsum("il,jk->ijkl", kd, kd) - trace_map
)
sym_tensor = jnp.einsum("ijkl,kl->ij", sym_map, A)
print(sym_tensor)
assert jnp.allclose(sym_tensor_actual, sym_tensor)

assert jnp.allclose(jnp.einsum("ijkl,ijkl", trace_map, antisym_map), 0.0)
assert jnp.allclose(jnp.einsum("ijkl,ijkl", trace_map, sym_map), 0.0)
assert jnp.allclose(jnp.einsum("ijkl,ijkl", antisym_map, sym_map), 0.0)

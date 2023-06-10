import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import imageio

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import jax.nn

import geometricconvolutions.geometric as geom
import geometricconvolutions.utils as utils

# ------------------------------------------------------------------------------
# Data generation functions

# Generate data for the gravity problem

def get_gravity_vector(position1, position2, mass):
    r_vec = position1 - position2
    r_squared = np.linalg.norm(r_vec) ** 3
    return (mass / r_squared) * r_vec

def get_gravity_field_image(N, D, point_position, point_mass):
    field = np.zeros((N,)*D + (D,))

    # this could all be vectorized
    for position in it.product(range(N), repeat=D):
        position = np.array(position)
        if (np.all(position == point_position)):
            continue

        field[tuple(position)] = get_gravity_vector(point_position, position, point_mass)

    return geom.GeometricImage(field, 0, D, is_torus=False)

def get_gravity_data(N, D, num_points, rand_key, num_images=1):
    rand_key, subkey = random.split(rand_key)
    planets = random.uniform(subkey, shape=(num_points,))
    planets = planets / jnp.max(planets)

    masses = []
    gravity_fields = []
    for _ in range(num_images):
        point_mass = np.zeros((N,N))
        gravity_field = geom.GeometricImage.zeros(N=N, k=1, parity=0, D=D, is_torus=False)

        # Sample uniformly the cells
        rand_key, subkey = random.split(rand_key)
        possible_locations = np.array(list(it.product(range(N), repeat=D)))
        location_choices = random.choice(subkey, possible_locations, shape=(num_points,), replace=False, axis=0)
        for (x,y), mass in zip(location_choices, planets):
            point_mass[x,y] = mass
            gravity_field = gravity_field + get_gravity_field_image(N, D, np.array([x,y]), mass)

        masses.append(geom.GeometricImage(point_mass, 0, D, is_torus=False))
        gravity_fields.append(gravity_field)

    return geom.BatchLayer.from_images(masses), geom.BatchLayer.from_images(gravity_fields)


# Generate data for the moving charges problems

def get_initial_charges(num_charges, N, D, rand_key):
    return N*random.uniform(rand_key, shape=(num_charges, D))

def get_velocity_vector(loc, charge_loc, charge):
    vec = loc - charge_loc
    scaling = jnp.linalg.norm(vec) ** 3
    return (charge / scaling) * vec

def get_velocity_field(N, D, charges):
    pixel_idxs = jnp.array(list(it.product(range(N), repeat=D)), dtype=int)
    velocity_field = jnp.zeros((N,)*D + (D,))

    vmap_get_vv = vmap(get_velocity_vector, in_axes=(0, None, None)) #all locs, one charge

    for charge in charges:
        velocity_field = velocity_field + vmap_get_vv(pixel_idxs, charge, 1).reshape((N,)*D + (D,))

    return geom.GeometricImage(velocity_field, 0, D, is_torus=False)

def update_charges(charges, delta_t):
    get_net_velocity = vmap(get_velocity_vector, in_axes=(None, 0, None)) #single loc, all charges

    new_charges = []
    for i in range(len(charges)):
        velocities = get_net_velocity(charges[i], jnp.concatenate((charges[:i], charges[i+1:])), 1)
        assert velocities.shape == (len(charges) - 1, 2)
        net_velocity = jnp.sum(velocities, axis=0)
        assert net_velocity.shape == charges[i].shape == (2,)
        new_charges.append(charges[i] + delta_t * net_velocity)
    return jnp.stack(new_charges)

def Qtransform(vector_field, s):
    vector_field_norm = vector_field.norm()
    return geom.GeometricImage(
        (4*(jax.nn.sigmoid(vector_field_norm.data / s)-0.5)) / vector_field_norm.data, 
        0, 
        vector_field.D,
        is_torus=vector_field.is_torus,
    ) * vector_field

def get_charge_data(N, D, num_charges, num_steps, delta_t, s, rand_key, num_images=1, outfile=None, warmup_steps=0):
    assert (not outfile) or (num_images == 1)

    initial_fields = []
    final_fields = []
    for _ in range(num_images):
        rand_key, subkey = random.split(rand_key)
        # generate charges, generally in the center so that they don't repel off the grid
        charges = get_initial_charges(num_charges, N/2, D, subkey) + jnp.array([int(N/4)]*D)
        for i in range(warmup_steps):
            charges = update_charges(charges, delta_t)

        initial_fields.append(Qtransform(get_velocity_field(N, D, charges), s))
        if outfile:
            utils.plot_image(initial_fields[-1])
            plt.savefig(f'{outfile}_{0}.png')
            plt.close()

        for i in range(1,num_steps+1):
            charges = update_charges(charges, delta_t)
            if outfile:
                utils.plot_image(Qtransform(get_velocity_field(N, D, charges), s))
                plt.savefig(f'{outfile}_{i}.png')
                plt.close()

        if outfile:
            with imageio.get_writer(f'{outfile}.gif', mode='I') as writer:
                for i in range(num_steps+1):
                    image = imageio.imread(f'{outfile}_{i}.png')
                    writer.append_data(image)

        final_fields.append(Qtransform(get_velocity_field(N, D, charges), s))

    return initial_fields, final_fields


import sys
sys.path.insert(0,'src/geometricconvolutions/')

import yt
import numpy as np
import jax.numpy as jnp
import geometric as geom

# yt.load_sample()
data_dir = '../data/AM06/'
ds = yt.load(data_dir + 'AM06.out1.00400.athdf')
# print(ds.field_list)
# print(ds.derived_field_list)
# print(ds.domain_width.in_units('cm'))

# print(len(ds.index.grids))
# print(ds.index.grids[0]['gas', 'temperature'])
# print(ds.point([0,-1000,1500])['gas', 'temperature'])

p = yt.ProjectionPlot(ds, "y", ("gas", "temperature"))
p.save('../images/')

# print(ds.all_data()['gas', 'temperature'].shape)

# all_data = ds.smoothed_covering_grid(
#     level=4,
#     left_edge=ds.domain_left_edge,
#     dims=ds.domain_dimensions, #128^3
# )
# D = len(all_data.shape)

# gas_temp = geom.GeometricImage(jnp.array(all_data['gas', 'temperature']), 0, D)

# operators = geom.make_all_operators(D)
# invariant_filters = geom.get_unique_invariant_filters(3,0,0,D,operators)

# print('before')
# gas_temp.convolve_with(invariant_filters[0])
# print('after')



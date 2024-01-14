import jax.numpy as jnp
from jax import random
import jax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

class TestMachineLearning:
    # Class to test the functions in the ml.py file, which include layers, data pre-processing, batching, etc.

    def testGetTimeSeriesXY(self):
        N = 5
        parity = 0
        D = 2
        ts = [geom.GeometricImage.fill(N, parity, D, fill_val) for fill_val in jnp.arange(10)]

        # regular
        X,Y = ml.get_timeseries_XY(ts, loss_steps=1, circular=False)
        assert len(X) == len(Y) == (len(ts)-1)
        for i in range(len(ts)-1):
            assert ts[i] == X[i]
            if i > 0:
                assert X[i] == Y[i-1]

        # loss_steps = 3
        X,Y = ml.get_timeseries_XY(ts, loss_steps=3, circular=False)
        assert X[0].__class__ == geom.GeometricImage
        assert isinstance(Y[0], list)
        assert len(Y[2]) == 3
        assert Y[3][0] == ts[4]
        assert Y[3][1] == ts[5]
        assert Y[3][2] == ts[6]
        assert Y[5][1] == Y[6][0]
        assert len(X) == len(Y) == (len(ts)-3)

        # circular
        X,Y = ml.get_timeseries_XY(ts, loss_steps=1, circular=True)
        assert len(X) == len(Y) == len(ts)
        assert X[-1] == ts[-1]
        assert Y[-1] == X[0]

        #circular, loss_steps=2
        X,Y = ml.get_timeseries_XY(ts, loss_steps=2, circular=True)
        assert len(X) == len(Y) == len(ts)
        assert len(Y[3]) == 2
        assert Y[-1][0] == X[0]
        assert Y[-1][1] == X[1]
        assert Y[-2][0] == X[-1]
        assert Y[-2][1] == X[0]

    def testGetBatchLayer(self):
        num_devices = 1 # since it can only see the cpu
        cpu = [jax.devices('cpu')[0]]
        key = random.PRNGKey(0)
        N = 5
        D = 2
        k = 0

        X = geom.BatchLayer({ (k,0): random.normal(key, shape=((10,1) + (N,)*D + (D,)*k)) }, D)
        Y = geom.BatchLayer({ (k,0): random.normal(key, shape=((10,1) + (N,)*D + (D,)*k)) }, D)

        batch_size = 2
        X_batches, Y_batches = ml.get_batch_layer(X, Y, batch_size=batch_size, rand_key=key, devices=cpu)
        assert len(X_batches) == len(Y_batches) == 5
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert X_batch[(k,0)].shape == Y_batch[(k,0)].shape == (num_devices, batch_size, 1) + (N,)*D + (D,)*k

        X = geom.BatchLayer(
            { 
                (0,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*1)),
            },
            D,
        )
        Y = geom.BatchLayer(
            { 
                (0,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*1)),
            },
            D,
        )

        # batching when the layer has multiple channels at different values of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batch_layer(X, Y, batch_size=batch_size, rand_key=key, devices=cpu)
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert X_batch[(0,0)].shape == Y_batch[(0,0)].shape == (num_devices, batch_size, 1) + (N,)*D + (D,)*0
            assert X_batch[(1,0)].shape == Y_batch[(1,0)].shape == (num_devices, batch_size, 1) + (N,)*D + (D,)*1

        X = geom.BatchLayer(
            { 
                (0,0): random.normal(key, shape=((20,2) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*1)),
            },
            D,
        )
        Y = geom.BatchLayer(
            { 
                (0,0): random.normal(key, shape=((20,2) + (N,)*D + (D,)*0)),
                (1,0): random.normal(key, shape=((20,1) + (N,)*D + (D,)*1)),
            },
            D,
        )

        # batching when layer has multiple channels for one value of k
        batch_size = 5
        X_batches, Y_batches = ml.get_batch_layer(X, Y, batch_size=batch_size, rand_key=key, devices=cpu)
        assert len(X_batches) == len(Y_batches) == 4
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert X_batch[(0,0)].shape == Y_batch[(0,0)].shape == (num_devices, batch_size, 2) + (N,)*D + (D,)*0
            assert X_batch[(1,0)].shape == Y_batch[(1,0)].shape == (num_devices, batch_size, 1) + (N,)*D + (D,)*1
            
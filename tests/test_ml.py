import jax.numpy as jnp
from jax import random

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

    def testGetBatchRollout(self):
        key = random.PRNGKey(0)
        N = 5
        parity = 0
        D = 2
        ts = [geom.GeometricImage.fill(N, parity, D, fill_val) for fill_val in jnp.arange(11)]

        X,Y = ml.get_timeseries_XY(ts, loss_steps=1, circular=False)

        batch_size = 2
        X_batches, Y_batches = ml.get_batch_rollout(X, Y, batch_size=batch_size, rand_key=key)
        assert len(X_batches) == len(Y_batches) == 5
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            assert X_batch.L == Y_batch.L == 2
            assert (X_batch + geom.BatchGeometricImage.fill(N, parity, D, 1, batch_size)) == Y_batch


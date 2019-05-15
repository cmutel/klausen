from klausen import NamedParameters
import numpy
import pytest
import stats_arrays as sa


def test_basic_example():
    first = {
        'bar': {
            'loc': 2,
            'scale': 0.5,
            'uncertainty_type': sa.NormalUncertainty.id
        }
    }
    second = {
        'foo': {
            'uncertainty_type': sa.TriangularUncertainty.id,
            'minimum': 0,
            'loc': 1,
            'maximum': 2
        }
    }
    np = NamedParameters(first)
    np.add_parameters(second)
    with pytest.raises(KeyError):
        np['foo']
    np.static()
    assert np['foo'] == 1
    assert np['bar'] == 2

    assert [x for x in np] == ['bar', 'foo']
    assert len(np) == 2

    with pytest.raises(NotImplementedError):
        np['x'] = 'y'

    np.stochastic(10)
    assert np['foo'].shape == (10,)
    assert 2 < np['foo'].sum() < 18

    np.stochastic(1)
    assert np['foo'].shape == (1,)

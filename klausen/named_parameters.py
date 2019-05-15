from collections.abc import Mapping
import numpy as np
import stats_arrays as sa
import warnings


class NamedParameters(Mapping):
    def __init__(self, params=None):
        self.data = {}
        self.metadata = {}
        self.values = {}
        if params:
            self.add_parameters(params)

    def __getitem__(self, key):
        TEXT = "No calculated values found; run `.static` or `.stochastic` first"
        if not self.values:
            warnings.warn(TEXT)
        return self.values[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("Use `.add_parameters` to add new parameters")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def add_parameters(self, params):
        for key, value in params.items():
            self.metadata[key] = value.pop('metadata', {})
            self.data[key] = value

    def _get_amount(self, dct):
        if 'amount' in dct:
            return dct['amount']
        elif dct.get('kind') in ('distribution', None):
            dist = sa.uncertainty_choices[dct['uncertainty_type']]
            median = float(dist.ppf(
                dist.from_dicts(dct),
                np.array((0.5,))
            ))
            dct['amount'] = median
            return median

    def static(self):
        # Stats_arrays parameters
        keys = sorted([key for key in self.data
                       if self.data[key].get('kind') in ('distribution', None)])
        self.values = {key: self._get_amount(self.data[key])
                       for key in keys}

    def stochastic(self, iterations=1000):
        # Stats_arrays parameters
        keys = sorted([key for key in self.data
                       if self.data[key].get('kind') in ('distribution', None)])
        array = sa.UncertaintyBase.from_dicts(*[self.data[key] for key in keys])
        rng = sa.MCRandomNumberGenerator(array)
        self.values = {key: row.reshape((-1,)) for key, row in zip(keys, rng.generate(iterations))}

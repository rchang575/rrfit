import numpy as np
from lmfit import Parameters

from rrfit.QTLS0fit import _store_qtls0_fit_summary
from rrfit.dataio import Device


class DummyResult:
    def __init__(self, params):
        self.params = params


def test_store_qtls0_fit_summary_sets_summary_compatible_attrs():
    params = Parameters()
    params.add("qtls0", value=1.2e6)
    params.add("nc", value=10.0)
    params.add("beta", value=0.7)
    params.add("Qother", value=5e6)
    params["qtls0"].stderr = 1.0e5
    params["nc"].stderr = None
    result = DummyResult(params)
    device = Device(name="device")

    _store_qtls0_fit_summary(device, result)

    assert device.qtls0_fit_result is result
    assert device.qtls0_best_values["qtls0"] == 1.2e6
    assert device.qtls0_best_values["Qother"] == 5e6
    assert device.qtls0_param_errors["qtls0"] == 1.0e5
    assert np.isnan(device.qtls0_param_errors["nc"])

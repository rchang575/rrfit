from pathlib import Path

import h5py
import numpy as np

from rrfit.dataio import Device, load_data


def _write_trace_file(path: Path, device_name: str):
    with h5py.File(path, "w") as file:
        file.attrs["device_name"] = device_name
        file.attrs["input_power"] = -30.0
        file.attrs["temp_avg"] = 0.025
        file.attrs["temp_std"] = 0.001
        file.attrs["tau"] = 1.2e-9
        file.create_dataset("frequency", data=np.array([1.0, 2.0, 3.0]))
        file.create_dataset("s21real", data=np.array([0.1, 0.2, 0.3]))
        file.create_dataset("s21imag", data=np.array([-0.1, -0.2, -0.3]))


def test_device_uses_independent_trace_lists():
    device_a = Device(name="a")
    device_b = Device(name="b")

    device_a.traces.append("sentinel")

    assert device_a.traces == ["sentinel"]
    assert device_b.traces == []


def test_load_data_appends_trace_when_device_traces_not_preinitialized(tmp_path):
    _write_trace_file(tmp_path / "trace.h5", "device_a")
    device = Device(name="device_a")

    load_data(tmp_path, device_a=device)

    assert len(device.traces) == 1
    trace = device.traces[0]
    assert trace.id == 0
    assert trace.device_name == "device_a"
    assert np.array_equal(trace.frequency, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(trace.s21real, np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(trace.s21imag, np.array([-0.1, -0.2, -0.3]))
    assert trace.power == -30.0
    assert trace.temperature == 0.025


def test_load_data_ignores_files_for_unknown_devices(tmp_path):
    _write_trace_file(tmp_path / "trace.h5", "device_a")
    device = Device(name="device_b")

    load_data(tmp_path, device_b=device)

    assert device.traces == []

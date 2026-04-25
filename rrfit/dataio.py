""" """

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


@dataclass
class Trace:
    id: Optional[int] = None
    device_name: Optional[str] = None
    frequency: Optional[np.ndarray] = None
    s21real: Optional[np.ndarray] = None
    s21imag: Optional[np.ndarray] = None
    power: Optional[float] = None
    temperature: Optional[float] = None
    temperature_err: Optional[float] = None
    background_amp: Optional[float] = None
    background_phase: Optional[float] = None
    tau: Optional[float] = None
    fr: Optional[float] = None
    fr_err: Optional[float] = None
    Qi: Optional[float] = None
    Qi_err: Optional[float] = None
    Ql: Optional[float] = None
    Ql_err: Optional[float] = None
    absQc: Optional[float] = None
    absQc_err: Optional[float] = None
    phi: Optional[float] = None
    phi_err: Optional[float] = None
    is_excluded: Optional[bool] = None
    is_homophasal: Optional[bool] = None


@dataclass
class Device:
    name: Optional[str] = None
    pitch: Optional[float] = None
    traces: list[Trace] = field(default_factory=list)
    line_attenuation: Optional[float] = None


def load_data(*folders: Path, **devices: Device):
    """ """
    for folder in folders:
        for path in Path(folder).iterdir():
            if not path.suffix in (".h5", ".hdf5", ".hdf"):
                continue

            with h5py.File(path) as file:
                device_name = file.attrs["device_name"]
                if device_name in devices:
                    trace = Trace(
                        device_name=device_name,
                        frequency=file["frequency"][:],
                        s21real=file["s21real"][:],
                        s21imag=file["s21imag"][:],
                        power=file.attrs.get("input_power"),
                        temperature=file.attrs.get("temp_avg"),
                        temperature_err=file.attrs.get("temp_std"),
                        background_amp=file.attrs.get("background_amp"),
                        background_phase=file.attrs.get("background_phase"),
                        tau=file.attrs.get("tau"),
                        fr=file.attrs.get("fr"),
                        fr_err=file.attrs.get("fr_err"),
                        Qi=file.attrs.get("Qi"),
                        Qi_err=file.attrs.get("Qi_err"),
                        Ql=file.attrs.get("Ql"),
                        Ql_err=file.attrs.get("Ql_err"),
                        absQc=file.attrs.get("absQc"),
                        absQc_err=file.attrs.get("absQc_err"),
                        phi=file.attrs.get("phi"),
                        phi_err=file.attrs.get("phi_err"),
                        is_homophasal=file.attrs.get("do_homophasal"),
                    )
                    devices[device_name].traces.append(trace)

    for device in devices.values():
        print(f"Found {len(device.traces)} traces for device '{device.name}'")
        for idx, trace in enumerate(device.traces):
            trace.id = idx

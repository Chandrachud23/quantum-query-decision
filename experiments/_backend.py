"""Resolve a fake or real IBM backend and build a matching sampler.

Credentials come from env (IBM_QUANTUM_TOKEN, IBM_CLOUD_INSTANCE) or
configs/hardware.py (gitignored); see configs/hardware.py.example.
"""

from __future__ import annotations

import os


def load_fake(name: str):
    import importlib
    mod = importlib.import_module("qiskit_ibm_runtime.fake_provider")
    return getattr(mod, name)()


def _load_credentials():
    """Return (token, instance, default_backend) from env or configs/hardware.py."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_CLOUD_INSTANCE")
    default_backend = "ibm_fez"
    if not token:
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import configs.hardware as h
            token = getattr(h, "IBM_QUANTUM_TOKEN", None)
            instance = instance or getattr(h, "IBM_CLOUD_INSTANCE", None)
            default_backend = getattr(h, "HARDWARE_BACKEND", default_backend)
        except Exception:
            pass
    if token == "YOUR_IBM_QUANTUM_TOKEN_HERE":
        token = None
    return token, instance, default_backend


def _connect_service():
    """QiskitRuntimeService, trying ibm_cloud then ibm_quantum_platform."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token, instance, _ = _load_credentials()
    if not token:                       # fall back to a previously saved account
        return QiskitRuntimeService()
    last = None
    for channel in ("ibm_cloud", "ibm_quantum_platform"):
        try:
            if channel == "ibm_cloud" and instance:
                return QiskitRuntimeService(channel=channel, token=token, instance=instance)
            return QiskitRuntimeService(channel=channel, token=token)
        except Exception as e:          # try the next channel
            last = e
    raise RuntimeError(f"Could not connect to IBM Quantum on any channel: {last}")


def resolve_backend(backend_name: str = "FakeFez", hardware: str | None = None,
                    min_qubits: int = 8):
    """Return (backend, is_hw). For hardware, `hardware` is a device name,
    'default' (use configs/hardware.py), or 'least_busy'."""
    if hardware:
        service = _connect_service()
        if hardware == "least_busy":
            backend = service.least_busy(operational=True, simulator=False,
                                         min_num_qubits=min_qubits)
        elif hardware == "default":
            backend = service.backend(_load_credentials()[2])
        else:
            backend = service.backend(hardware)
        return backend, True
    return load_fake(backend_name), False


def make_sampler(backend, is_hw: bool, seed: int = 1):
    """A SamplerV2 that runs on hardware, or on Aer with the backend's noise model."""
    if is_hw:
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
        return RuntimeSampler(mode=backend)
    from qiskit_aer.primitives import SamplerV2
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel.from_backend(backend)
    return SamplerV2(options={"backend_options": {"noise_model": nm}}, seed=seed)

"""Signal-flow solver for S-parameter RF networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from device import RFDevice


@dataclass(frozen=True)
class SignalState:
    """Solved power-wave phasors for an RF network."""

    incident: np.ndarray
    outgoing: np.ndarray
    source: np.ndarray
    load_reflection: np.ndarray

    @property
    def reflected_waves(self) -> np.ndarray:
        return self.outgoing

    @property
    def incident_power(self) -> np.ndarray:
        return np.abs(self.incident) ** 2

    @property
    def power_at_ports(self) -> np.ndarray:
        return self.outgoing_power

    @property
    def outgoing_power(self) -> np.ndarray:
        return np.abs(self.outgoing) ** 2

    @property
    def net_power_into_device(self) -> np.ndarray:
        return self.incident_power - self.outgoing_power


class NetworkSimulator:
    """Solve ``b = S @ a`` for an S-matrix RF device.

    ``source`` is the externally launched incident power wave at each port,
    and ``load_reflection`` contains each port's load reflection coefficient.
    Nonzero reflection coefficients feed some outgoing wave back into the
    device input, giving the steady-state equations:

        b = S @ a
        a = source + Gamma @ b

    Substitution gives:

        b = S @ source + S @ Gamma @ b
        (I - S @ Gamma) @ b = S @ source

    That linear solve includes all repeated reflections in one steady-state
    answer without explicitly simulating each bounce.
    """

    def __init__(self, device: RFDevice) -> None:
        self.device = device

    def propagate(
        self,
        source: Sequence[complex] | np.ndarray,
        load_reflection: Sequence[complex] | np.ndarray | None = None,
    ) -> SignalState:
        source_vector = self._as_port_vector(source, "source")
        gamma_vector = (
            np.zeros(self.device.port_count, dtype=np.complex128)
            if load_reflection is None
            else self._as_port_vector(load_reflection, "load_reflection")
        )

        gamma_matrix = np.diag(gamma_vector)
        system_matrix = np.eye(self.device.port_count, dtype=np.complex128)
        system_matrix -= self.device.s_matrix @ gamma_matrix

        # S @ source is the first pass through the network. The solved
        # reflected waves include every later bounce caused by Gamma.
        outgoing = np.linalg.solve(system_matrix, self.device.s_matrix @ source_vector)
        incident = source_vector + gamma_matrix @ outgoing

        return SignalState(
            incident=incident,
            outgoing=outgoing,
            source=source_vector,
            load_reflection=gamma_vector,
        )

    def propagate_from_port(
        self,
        port: int,
        amplitude: complex = 1.0 + 0.0j,
        load_reflection: Sequence[complex] | np.ndarray | None = None,
    ) -> SignalState:
        """Launch a phasor into one 1-based port and solve the network."""

        if port < 1 or port > self.device.port_count:
            raise ValueError(f"port must be in the range 1..{self.device.port_count}")

        source = np.zeros(self.device.port_count, dtype=np.complex128)
        source[port - 1] = amplitude
        return self.propagate(source, load_reflection=load_reflection)

    def solve_steady_state(
        self,
        source_port: int,
        amplitude: complex = 1.0 + 0.0j,
        load_reflection_coefficients: Sequence[complex] | np.ndarray | None = None,
    ) -> SignalState:
        """Solve reflected waves and port powers from one driven 1-based port."""

        return self.propagate_from_port(
            port=source_port,
            amplitude=amplitude,
            load_reflection=load_reflection_coefficients,
        )

    def _as_port_vector(
        self,
        values: Sequence[complex] | np.ndarray,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.complex128)
        if vector.shape != (self.device.port_count,):
            raise ValueError(
                f"{name} must be a 1D vector with {self.device.port_count} entries"
            )
        return vector

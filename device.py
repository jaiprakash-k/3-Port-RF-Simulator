"""RF device models built from scattering matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Direction = Literal["clockwise", "counterclockwise"]


@dataclass(frozen=True)
class CirculatorSpec:
    """Non-ideal 3-port circulator parameters.

    All dB values are positive engineering quantities:
    insertion_loss_db=0.5 means the intended through path has -0.5 dB gain,
    isolation_db=25 means reverse leakage is -25 dB, and return_loss_db=20
    means each matched-port reflection coefficient has magnitude -20 dB.
    """

    insertion_loss_db: float = 0.5
    isolation_db: float = 30.0
    return_loss_db: float = 20.0
    direction: Direction = "clockwise"
    phase_shift_deg: float = 0.0
    leakage_phase_shift_deg: float = 0.0
    reflection_phase_shift_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in ("clockwise", "counterclockwise"):
            raise ValueError("direction must be 'clockwise' or 'counterclockwise'")
        for name in ("insertion_loss_db", "isolation_db", "return_loss_db"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def db_to_magnitude(db_value: float) -> float:
    """Convert a dB loss/isolation/return-loss value to linear magnitude."""

    return float(10.0 ** (-db_value / 20.0))


def phasor(magnitude: float, phase_shift_deg: float = 0.0) -> complex:
    """Build a complex RF phase-shift phasor using ``magnitude * e^(-j theta)``."""

    return complex(magnitude * np.exp(-1j * np.deg2rad(phase_shift_deg)))


@dataclass(frozen=True)
class Circulator:
    """Three-port non-reciprocal RF circulator.

    The forward path is P1->P2, P2->P3, and P3->P1 for the default clockwise
    orientation. Reverse paths are populated with low-magnitude isolation
    leakage, and diagonal terms model finite return loss.
    """

    insertion_loss_db: float = 0.5
    isolation_db: float = 30.0
    return_loss_db: float = 20.0
    phase_shift_deg: float = 0.0
    leakage_phase_shift_deg: float = 0.0
    reflection_phase_shift_deg: float = 0.0
    direction: Direction = "clockwise"

    def __post_init__(self) -> None:
        if self.direction not in ("clockwise", "counterclockwise"):
            raise ValueError("direction must be 'clockwise' or 'counterclockwise'")
        for name in ("insertion_loss_db", "isolation_db", "return_loss_db"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_spec(cls, spec: CirculatorSpec) -> "Circulator":
        return cls(
            insertion_loss_db=spec.insertion_loss_db,
            isolation_db=spec.isolation_db,
            return_loss_db=spec.return_loss_db,
            phase_shift_deg=spec.phase_shift_deg,
            leakage_phase_shift_deg=spec.leakage_phase_shift_deg,
            reflection_phase_shift_deg=spec.reflection_phase_shift_deg,
            direction=spec.direction,
        )

    def s_matrix(self) -> np.ndarray:
        """Return the circulator 3x3 complex S-matrix as a NumPy array."""

        through = phasor(db_to_magnitude(self.insertion_loss_db), self.phase_shift_deg)
        leakage = phasor(
            db_to_magnitude(self.isolation_db),
            self.leakage_phase_shift_deg,
        )
        reflection = phasor(
            db_to_magnitude(self.return_loss_db),
            self.reflection_phase_shift_deg,
        )

        s_matrix = np.zeros((3, 3), dtype=np.complex128)
        np.fill_diagonal(s_matrix, reflection)

        if self.direction == "clockwise":
            forward_paths = ((0, 1), (1, 2), (2, 0))
        else:
            forward_paths = ((0, 2), (2, 1), (1, 0))

        for input_port, output_port in forward_paths:
            s_matrix[output_port, input_port] = through
            s_matrix[input_port, output_port] = leakage

        return s_matrix


def build_circulator_s_matrix(spec: CirculatorSpec | None = None) -> np.ndarray:
    """Build a 3x3 S-matrix for a non-ideal circulator.

    The S-matrix uses the standard power-wave convention ``b = S @ a``:
    ``a[j]`` is the incident wave at port j and ``b[i]`` is the outgoing wave
    at port i. Indices in this module are zero-based, so S[1, 0] is S21.
    """

    return Circulator.from_spec(spec or CirculatorSpec()).s_matrix()


@dataclass(frozen=True)
class RFDevice:
    """Generic RF N-port device represented by an S-matrix."""

    s_matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.s_matrix, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("s_matrix must be a square 2D array")
        object.__setattr__(self, "s_matrix", matrix)

    @property
    def port_count(self) -> int:
        return int(self.s_matrix.shape[0])

    @classmethod
    def circulator(cls, spec: CirculatorSpec | None = None) -> "RFDevice":
        return cls(build_circulator_s_matrix(spec))

"""RF metrics derived from S-parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOG_EPSILON = 1e-15


@dataclass(frozen=True)
class PortMetrics:
    input_port: int
    through_port: int
    isolated_port: int
    insertion_loss_db: float
    isolation_db: float
    return_loss_db: float


def magnitude_to_db(magnitude: float) -> float:
    """Convert a linear complex-wave magnitude to dB."""

    magnitude = max(float(abs(magnitude)), LOG_EPSILON)
    return float(20.0 * np.log10(magnitude))


def loss_db_from_magnitude(magnitude: float) -> float:
    """Convert a linear magnitude below unity to a positive loss in dB."""

    return -magnitude_to_db(magnitude)


def insertion_loss_db(s_matrix: np.ndarray, input_port: int, output_port: int) -> float:
    matrix = _validated_s_matrix(s_matrix)
    _validate_port(input_port, matrix.shape[0])
    _validate_port(output_port, matrix.shape[0])
    return loss_db_from_magnitude(matrix[output_port - 1, input_port - 1])


def isolation_db(s_matrix: np.ndarray, input_port: int, isolated_port: int) -> float:
    matrix = _validated_s_matrix(s_matrix)
    _validate_port(input_port, matrix.shape[0])
    _validate_port(isolated_port, matrix.shape[0])
    return loss_db_from_magnitude(matrix[isolated_port - 1, input_port - 1])


def return_loss_db(s_matrix: np.ndarray, port: int) -> float:
    matrix = _validated_s_matrix(s_matrix)
    _validate_port(port, matrix.shape[0])
    return loss_db_from_magnitude(matrix[port - 1, port - 1])


def circulator_port_metrics(s_matrix: np.ndarray) -> list[PortMetrics]:
    """Return per-port metrics for a clockwise P1->P2->P3->P1 circulator."""

    matrix = _validated_s_matrix(s_matrix)
    if matrix.shape != (3, 3):
        raise ValueError("circulator metrics require a 3x3 S-matrix")

    topology = (
        (1, 2, 3),
        (2, 3, 1),
        (3, 1, 2),
    )

    return [
        PortMetrics(
            input_port=input_port,
            through_port=through_port,
            isolated_port=isolated_port,
            insertion_loss_db=insertion_loss_db(matrix, input_port, through_port),
            isolation_db=isolation_db(matrix, input_port, isolated_port),
            return_loss_db=return_loss_db(matrix, input_port),
        )
        for input_port, through_port, isolated_port in topology
    ]


def rf_performance_report(
    s_matrix: np.ndarray,
    reflected_waves: Sequence[complex] | np.ndarray,
    incident_waves: Sequence[complex] | np.ndarray | None = None,
    source_waves: Sequence[complex] | np.ndarray | None = None,
) -> dict[str, object]:
    """Return RF performance metrics in dictionary format.

    S-parameter losses use ``-20 log10(|Sij|)`` with a small floor so ideal
    zeros do not produce ``log(0)``. Power waves use normalized RF power,
    so power is simply ``|wave|^2``.
    """

    matrix = _validated_s_matrix(s_matrix)
    port_count = matrix.shape[0]
    b_waves = _validated_port_vector(reflected_waves, port_count, "reflected_waves")

    incident = (
        None
        if incident_waves is None
        else _validated_port_vector(incident_waves, port_count, "incident_waves")
    )
    source = (
        None
        if source_waves is None
        else _validated_port_vector(source_waves, port_count, "source_waves")
    )

    output_powers = np.abs(b_waves) ** 2
    total_output_power = float(np.sum(output_powers))

    if incident is not None:
        input_power = float(np.sum(np.abs(incident) ** 2))
    elif source is not None:
        input_power = float(np.sum(np.abs(source) ** 2))
    else:
        input_power = 0.0

    efficiency = (
        total_output_power / input_power
        if input_power > LOG_EPSILON
        else 0.0
    )

    return {
        "isolation_db": _isolation_dict(matrix),
        "return_loss_db": _return_loss_dict(matrix),
        "output_power_per_port": _port_value_dict(output_powers),
        "total_output_power": total_output_power,
        "input_power_reference": input_power,
        "efficiency": float(efficiency),
        "efficiency_percent": float(100.0 * efficiency),
    }


def _validated_s_matrix(s_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(s_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("s_matrix must be a square 2D array")
    return matrix


def _validated_port_vector(
    values: Sequence[complex] | np.ndarray,
    port_count: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.shape != (port_count,):
        raise ValueError(f"{name} must be a 1D vector with {port_count} entries")
    return vector


def _validate_port(port: int, port_count: int) -> None:
    if port < 1 or port > port_count:
        raise ValueError(f"port must be in the range 1..{port_count}")


def _safe_loss_db(value: complex) -> float:
    return -20.0 * float(np.log10(max(abs(value), LOG_EPSILON)))


def _isolation_dict(s_matrix: np.ndarray) -> dict[str, float]:
    isolation: dict[str, float] = {}
    port_count = s_matrix.shape[0]
    for output_port in range(port_count):
        for input_port in range(port_count):
            if output_port == input_port:
                continue
            key = f"S{output_port + 1}{input_port + 1}"
            isolation[key] = _safe_loss_db(s_matrix[output_port, input_port])
    return isolation


def _return_loss_dict(s_matrix: np.ndarray) -> dict[str, float]:
    return {
        f"S{port + 1}{port + 1}": _safe_loss_db(s_matrix[port, port])
        for port in range(s_matrix.shape[0])
    }


def _port_value_dict(values: np.ndarray) -> dict[str, float]:
    return {
        f"P{port}": float(value)
        for port, value in enumerate(values, start=1)
    }

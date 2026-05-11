"""Run a basic 3-port non-reciprocal RF network simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from device import Circulator, RFDevice
from metrics import circulator_port_metrics, rf_performance_report
from simulator import NetworkSimulator
from visualization import plot_output_power_bar, plot_s_matrix_heatmap

OUTPUT_DIR = Path("outputs")
SOURCE_PORT = 1
SOURCE_AMPLITUDE = 1.0 + 0.0j


def create_circulator() -> RFDevice:
    """Build the RF circulator device used by the simulation."""

    circulator = Circulator(
        insertion_loss_db=0.6,
        isolation_db=32.0,
        return_loss_db=22.0,
        direction="clockwise",
        phase_shift_deg=8.0,
        leakage_phase_shift_deg=-35.0,
        reflection_phase_shift_deg=-180.0,
    )
    return RFDevice(circulator.s_matrix())


def run_port_1_simulation(device: RFDevice):
    """Inject a source at Port 1 and solve steady-state waves."""

    simulator = NetworkSimulator(device)
    load_gamma = np.array(
        [0.05 + 0.00j, 0.02 - 0.01j, 0.03 + 0.02j],
        dtype=np.complex128,
    )
    return simulator.solve_steady_state(
        source_port=SOURCE_PORT,
        amplitude=SOURCE_AMPLITUDE,
        load_reflection_coefficients=load_gamma,
    )


def print_power_results(power_by_port: dict[str, float], total_power: float) -> None:
    print("\nOutput power:")
    for port, power in power_by_port.items():
        print(f"{port}: {power:.6f}")
    print(f"Total output power: {total_power:.6f}")


def print_circulator_metrics(device: RFDevice) -> None:
    print("\nIsolation and return loss:")
    for item in circulator_port_metrics(device.s_matrix):
        print(
            f"P{item.input_port}: "
            f"isolation to P{item.isolated_port} = {item.isolation_db:.2f} dB, "
            f"return loss = {item.return_loss_db:.2f} dB"
        )


def generate_plots(device: RFDevice, output_power: np.ndarray) -> None:
    heatmap_path = OUTPUT_DIR / "s_matrix_heatmap.png"
    power_path = OUTPUT_DIR / "output_power_bar.png"

    plot_s_matrix_heatmap(device.s_matrix, output_path=heatmap_path)
    plot_output_power_bar(output_power, output_path=power_path)

    print("\nGenerated plots:")
    print(f"S-matrix heatmap: {heatmap_path}")
    print(f"Output power bar chart: {power_path}")


def main() -> None:
    # 1. Create a realistic non-ideal clockwise circulator.
    device = create_circulator()

    # 2-3. Inject a complex phasor at Port 1 and solve the steady-state network.
    state = run_port_1_simulation(device)

    report = rf_performance_report(
        s_matrix=device.s_matrix,
        reflected_waves=state.reflected_waves,
        incident_waves=state.incident,
        source_waves=state.source,
    )

    print("S-matrix:")
    print(np.array2string(device.s_matrix, precision=5, suppress_small=False))

    # 4. Print the requested RF results.
    print_power_results(
        power_by_port=report["output_power_per_port"],
        total_power=report["total_output_power"],
    )
    print_circulator_metrics(device)
    print(f"\nEfficiency: {report['efficiency_percent']:.2f}%")

    # 5. Generate visualization files.
    generate_plots(device, state.power_at_ports)


if __name__ == "__main__":
    main()

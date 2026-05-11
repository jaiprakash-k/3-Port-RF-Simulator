# qthack — 3‑Port Non‑Reciprocal RF Network Simulation

A small, self-contained Python project that models a **non-ideal 3-port RF circulator** using an **S-parameter (scattering matrix)** representation, solves the **steady-state power-wave response** under arbitrary port mismatches, and computes common RF metrics (insertion loss, isolation, return loss) plus basic power/efficiency reporting.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Metrics](#metrics)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Project Structure](#project-structure)

## Overview

This repo simulates a 3-port circulator with realistic non-idealities:

- **Through path loss** (insertion loss)
- **Reverse leakage** (finite isolation)
- **Finite match** at each port (return loss)
- **Phase shifts** applied to through/leakage/reflection terms

The default run:
- Builds a clockwise circulator device
- Injects a complex source wave into **Port 1**
- Applies small nonzero load reflection coefficients (mismatches) on all ports
- Solves steady state and prints:
  - Per-port output powers and total output power
  - Isolation and return loss per input port
  - An overall efficiency estimate

## Architecture

Data flow in a typical run:

1. `device.Circulator` builds a non-ideal 3×3 complex S-matrix
2. `simulator.NetworkSimulator` solves the steady-state power-wave equations
3. `metrics.rf_performance_report` and `metrics.circulator_port_metrics` compute dB metrics and power summaries
4. `visualization` generates plots to disk using a headless Matplotlib backend

## How It Works

### Power-wave steady-state solve

The simulator uses the standard power-wave convention:

- $\mathbf{b} = \mathbf{S}\,\mathbf{a}$

and models load mismatch using a diagonal reflection matrix $\Gamma$:

- $\mathbf{a} = \mathbf{a}_{\text{source}} + \Gamma\,\mathbf{b}$

Substituting and solving gives a single linear system that implicitly includes all repeated reflections:

- $(\mathbf{I} - \mathbf{S}\Gamma)\,\mathbf{b} = \mathbf{S}\,\mathbf{a}_{\text{source}}$

This is implemented in `NetworkSimulator.propagate()`.

### Circulator S-matrix model

The circulator is constructed as a 3×3 complex S-matrix:

- Off-diagonal “forward” terms implement the intended circulation direction (clockwise or counterclockwise)
- Reverse terms implement finite isolation leakage
- Diagonal terms implement finite return loss

See `Circulator.s_matrix()`.

## Metrics

The project reports common RF quantities derived from S-parameters:

- **Insertion loss** (positive dB): $\mathrm{IL}_{ij} = -20\log_{10}(|S_{ij}|)$
- **Isolation** (positive dB): computed the same way on the “should-be-blocked” path
- **Return loss** (positive dB): $\mathrm{RL}_{ii} = -20\log_{10}(|S_{ii}|)$

Power is computed from normalized power-waves, so per-port output power is simply $|b_i|^2$.

## Quick Start

### Prerequisites

- Python **3.10+**

### Install dependencies

This project is a few pure-Python files; install the runtime deps into a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

pip install numpy matplotlib
```

Optional (only needed if you want to save animations via Pillow):

```bash
pip install pillow
```

### Run

```bash
python3 main.py
```

The script prints the S-matrix and computed metrics, and writes plot images into `outputs/`.

## Configuration

The simplest knobs live in `create_circulator()` in `main.py`:

- `insertion_loss_db`: intended forward-path loss (positive dB value)
- `isolation_db`: reverse leakage attenuation (positive dB value)
- `return_loss_db`: port match quality (positive dB value)
- `direction`: `"clockwise"` or `"counterclockwise"`
- `phase_shift_deg`, `leakage_phase_shift_deg`, `reflection_phase_shift_deg`

You can also change the load mismatch vector in `run_port_1_simulation()`:

- `load_gamma`: per-port complex reflection coefficient $\Gamma_i$

## Outputs

A default run generates:

- `outputs/s_matrix_heatmap.png` — heatmap of $20\log_{10}|S_{ij}|$
- `outputs/output_power_bar.png` — bar chart of normalized per-port output power $|b_i|^2$

The plotting code forces a headless Matplotlib backend (`Agg`) so it can run without a GUI.

## Project Structure

- `main.py` — entrypoint that builds the device, runs the solve, prints metrics, and generates plots
- `device.py` — circulator and generic `RFDevice` S-matrix models
- `simulator.py` — steady-state signal-flow solver (`NetworkSimulator`) and `SignalState`
- `metrics.py` — helper functions for insertion loss / isolation / return loss and summary reporting
- `visualization.py` — plotting helpers (S-matrix heatmap, output power chart, optional animation)
- `outputs/` — generated artifacts (plots)

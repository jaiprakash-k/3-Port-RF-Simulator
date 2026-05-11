# qthack — 3‑Port Non‑Reciprocal RF Network Simulation

A self-contained Python project that models a **non-ideal 3-port RF circulator** using an **S-parameter (scattering matrix)** representation, solves the **steady-state power-wave response** under arbitrary port mismatches, and computes common RF metrics (insertion loss, isolation, return loss) along with power/efficiency reporting.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Theory & Math](#theory--math)
- [Code Walkthrough](#code-walkthrough)
  - [`device.py` — S-Matrix Model](#devicepy--s-matrix-model)
  - [`simulator.py` — Steady-State Solver](#simulatorpy--steady-state-solver)
  - [`metrics.py` — RF Performance Metrics](#metricspy--rf-performance-metrics)
  - [`visualization.py` — Plotting](#visualizationpy--plotting)
  - [`main.py` — Entrypoint](#mainpy--entrypoint)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Project Structure](#project-structure)

---

## Overview

This repo simulates a 3-port circulator with realistic non-idealities:

- **Through path loss** (insertion loss)
- **Reverse leakage** (finite isolation)
- **Finite match** at each port (return loss)
- **Phase shifts** applied to through/leakage/reflection terms

A default run:
1. Builds a clockwise circulator device
2. Injects a complex source wave into **Port 1**
3. Applies small nonzero load reflection coefficients (mismatches) on all ports
4. Solves steady state and prints:
   - Per-port output powers and total output power
   - Isolation and return loss per input port
   - Overall efficiency estimate

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌────────────────┐
│ device.py   │ ──▶ │ simulator.py     │ ──▶ │ metrics.py   │ ──▶ │ visualization  │
│ S-matrix    │     │ Steady-state     │     │ IL/Iso/RL    │     │ Plots → PNG    │
└─────────────┘     │ (I − SΓ)b = Sa   │     └──────────────┘     └────────────────┘
                    └──────────────────┘
```

---

## Theory & Math

### Power-Wave Convention

Each port has an **incident wave** $a_i$ and a **reflected wave** $b_i$. The S-matrix relates them:

$$\mathbf{b} = \mathbf{S}\,\mathbf{a}$$

Power flowing **out** of port $i$ is $|b_i|^2$ (normalized).

### Load Mismatch via Reflection Matrix

A non-ideal load at port $i$ reflects part of $b_i$ back as additional incident wave. With a diagonal reflection matrix $\Gamma = \text{diag}(\Gamma_1, \Gamma_2, \Gamma_3)$:

$$\mathbf{a} = \mathbf{a}_{\text{source}} + \Gamma\,\mathbf{b}$$

### Steady-State Solve

Substituting $\mathbf{a}$ into $\mathbf{b} = \mathbf{S}\mathbf{a}$:

$$\mathbf{b} = \mathbf{S}(\mathbf{a}_{\text{source}} + \Gamma\mathbf{b})$$

$$(\mathbf{I} - \mathbf{S}\Gamma)\,\mathbf{b} = \mathbf{S}\,\mathbf{a}_{\text{source}}$$

This single linear solve **implicitly captures all infinite re-reflections** between device and loads.

### Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Insertion Loss** | $\mathrm{IL}_{ij} = -20\log_{10}\|S_{ij}\|$ | Loss on intended path |
| **Isolation** | $\mathrm{Iso}_{ij} = -20\log_{10}\|S_{ij}\|$ | Attenuation on blocked path |
| **Return Loss** | $\mathrm{RL}_{ii} = -20\log_{10}\|S_{ii}\|$ | Port match quality |
    
---

## Quick Start

### Prerequisites
- Python **3.10+**

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

Optional (for animated GIFs):

```bash
pip install pillow
```

### Run

```bash
python3 main.py
```

Prints the S-matrix, computed metrics, and writes PNG plots into `outputs/`.

---

## Configuration

Tunable knobs in `create_circulator()` (in `main.py`):

| Parameter | Description |
|---|---|
| `insertion_loss_db` | Forward-path loss (positive dB) |
| `isolation_db` | Reverse leakage attenuation (positive dB) |
| `return_loss_db` | Port match quality (positive dB) |
| `direction` | `"clockwise"` or `"counterclockwise"` |
| `phase_shift_deg` | Phase on through path |
| `leakage_phase_shift_deg` | Phase on reverse leakage |
| `reflection_phase_shift_deg` | Phase on diagonal reflection |

In `run_port_1_simulation()` you can adjust:

- `a_source` — incident wave injection (which port, complex amplitude)
- `load_gamma` — complex reflection coefficient per port

---

## Outputs

A default run generates:

- `outputs/s_matrix_heatmap.png` — heatmap of $20\log_{10}|S_{ij}|$
- `outputs/output_power_bar.png` — bar chart of $|b_i|^2$ per port

---

## Project Structure

```
qthack/
├── main.py             # Entrypoint: builds device, runs solve, prints metrics, plots
├── device.py           # RFDevice / Circulator S-matrix models
├── simulator.py        # NetworkSimulator + SignalState (steady-state solver)
├── metrics.py          # IL / Isolation / Return Loss helpers + report
├── visualization.py    # Headless Matplotlib plotting helpers
└── outputs/            # Generated PNGs
```

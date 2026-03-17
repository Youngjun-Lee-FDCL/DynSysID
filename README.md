# GP-NARX with Multi-Experiment Training and Simulation-Run Validation

This repository implements a Gaussian Process NARX (GP-NARX) model trained on multiple experiments and evaluated not only on standard validation sets but also on independent simulation runs.

---

## 🔹 Motivation

Standard system identification workflows typically evaluate models using one-step-ahead prediction on held-out validation data. However, such evaluations can be overly optimistic since the model always uses true past outputs.

This project focuses on a more realistic and challenging evaluation:

> **Can the learned model reproduce system dynamics in a free-run (rollout) simulation?**

---

## 🔹 Method Overview

The pipeline consists of the following steps:

1. **Multi-experiment training**
   - Data from multiple independent experiments are aggregated
   - Optional block-coverage sampling is used to improve training diversity

2. **GP-NARX modeling**
   - Nonlinear mapping:
     ```
     y(k) = f(y(k-1), ..., u(k-1), ...)
     ```
   - One GP model per output channel

3. **Evaluation modes**
   - One-step-ahead prediction (standard)
   - Free-run simulation (rollout)

### Example: Nonlinear Aircraft PID Tracking

The dataset is generated from a nonlinear 6-state aircraft-like model controlled by PID loops.

- 3 outputs: pitch, roll, yaw rate  
- 4 inputs: control surfaces + throttle  
- Includes actuator saturation and process/measurement noise  

#### Reference tracking
The controller tracks piecewise-constant references with moderate nonlinear coupling.

#### Control inputs
Actuator signals show saturation and cross-coupling effects.

#### Why this dataset?
This example provides:
- Strong nonlinearity
- Multi-input multi-output coupling
- Realistic control-driven dynamics

![result](images/result_aircraft.png)

---

## 🚧 Status

- GP-NARX: Stable and fully functional
- VGP-SSM: Under development (not fully validated yet)
> ⚠️ VGP-SSM is currently under development. Results may be unstable.


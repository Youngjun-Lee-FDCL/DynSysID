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

---

## 🔹 Key Idea: Free-Run Simulation

In free-run mode, the model recursively feeds its own predictions:

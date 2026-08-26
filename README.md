# Label-Efficient Illicit Transaction Detection on the Elliptic Bitcoin Graph via Self-Supervised Pretraining

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official codebase for the Deep Learning Graduate Project, Department of Electrical Engineering, Sharif University of Technology.

---

## 📌 Overview
An end-to-end framework leveraging **Graph Self-Supervised Learning (SSL)** to detect illicit cryptocurrency transactions on the temporal **Elliptic Bitcoin Dataset**.

### Key Contributions:
* **Self-Supervised Pretraining (GRACE-style):** Node-level contrastive learning (InfoNCE) on the full graph topology to extract structural invariants.
* **Extension Track B1 (Knowledge Distillation):** Teacher-Student distillation transferring soft supervision from a fully supervised teacher to regularize representations under severe label scarcity (10% regime).
* **Extension Track A2 (Domain-Adversarial Training):** Gradient Reversal Layer (GRL) enforcing temporal domain invariance across the dark-market shutdown at time step 43.

---

## 📊 Benchmark Results

| Model | 100% Labels | 50% Labels | 25% Labels | 10% Labels | Pre-SD | Post-SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline** | 0.6046 | 0.5737 | 0.4952 | 0.4509 | 0.6840 | 0.0270 |
| **SSL (Fine-tuned)** | 0.5978 | 0.5701 | 0.5706 | 0.4211 | 0.6998 | 0.0567 |
| **SSL + A2 (Domain Adv)** | 0.5818 | 0.4757 | 0.5190 | 0.4320 | 0.6832 | 0.0349 |
| **SSL + B1 (Distillation)** | 0.6067 | 0.5552 | 0.5701 | 0.4763 | 0.6962 | 0.0228 |
| **SSL + A2 + B1** | 0.6019 | 0.5483 | 0.5518 | 0.4754 | 0.6861 | 0.0310 |

---

## 👥 Authors
* **Amirabbas Dashtbozorg** - Department of Electrical Engineering, Sharif University of Technology
* **Kianoush Kadkhodaie** - Department of Civil Engineering, Sharif University of Technology

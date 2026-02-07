# Machine Learning for Quantum State Tomography

## 1. Project Overview
This project integrates Machine Learning (ML) with fundamental quantum algorithms to address the "readout bottleneck" in quantum computing. We successfully demonstrated a complete workflow that:
1.  **Classifies Quantum Noise:** Rapidly identifies noise channels (Depolarizing vs. Amplitude Damping) using supervised learning.
2.  **Solves Linear Systems:** Implements the HHL algorithm for a $4 \times 4$ system.
3.  **Verifies Results:** Uses ML-enhanced tomography to reconstruct the solution state $|x\rangle$ and validate the solver's accuracy.

## 2. Methodology
### A. Efficient State Readout (Assignments 1-3)
We developed a scalable tomography pipeline. Instead of relying solely on expensive Maximum Likelihood Estimation (MLE), we trained neural networks (MLPs) to map measurement statistics directly to density matrices $\rho$.
* **Data:** Single-qubit Pauli measurements ($X, Y, Z$).
* **Model:** A custom `DensityMatrixMLP` that guarantees physical constraints (positivity, trace=1) via Cholesky decomposition.

### B. Calibration-Time Classification (Assignment 4)
To ensure hardware reliability, we built a classifier to tag noise environments.
* **Input:** Flattened Choi matrix elements of the quantum channel.
* **Algorithm:** Logistic Regression.
* **Result:** 95% accuracy in distinguishing between Depolarizing and Amplitude Damping errors.

### C. HHL Algorithm & Verification (Assignment 5)
We solved the linear system $A\vec{x} = \vec{b}$ where:
$$A = \begin{pmatrix} 1.5 & 0.2 & 0 & 0 \\ 0.2 & 1.3 & 0.1 & 0 \\ 0 & 0.1 & 1.2 & 0.1 \\ 0 & 0 & 0.1 & 1.1 \end{pmatrix}, \quad \vec{b} = \begin{pmatrix} 1.0 \\ 0.5 \\ 0.2 \\ 0.1 \end{pmatrix}$$

The quantum solution $|x_{HHL}\rangle$ was read out using our tomography pipeline and compared to the exact classical solution $x_{class} = A^{-1}b$.

## 3. Key Results
* **HHL Fidelity:** The verified quantum solution achieved a fidelity of **$F \approx 1.00$** against the classical baseline.
* **Tomography Robustness:** The reconstruction pipeline successfully recovered the state amplitudes with a residual norm of $O(10^{-16})$, confirming the validity of the HHL circuit execution.

## 4. Repository Structure
| Folder | Description |
| :--- | :--- |
| `data/` | Measurement datasets (.npy) for qubit states $|0\rangle, |1\rangle, |+\rangle$. |
| `models/` | Serialized checkpoints (.pkl) for the QST surrogate and classifier. |
| `notebooks/` | Executable Jupyter notebooks for Assignments 1-5. |
| `src/` | Modular Python scripts for model training (`train.py`) and metrics (`fidelity`). |
| `results/` | Final report and LaTeX appendices. |
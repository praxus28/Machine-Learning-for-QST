# Methodology

## Experiment Setup
A scalability experiment was conducted by increasing the number of qubits from 1 to 5.

For each qubit count:
- A random pure target quantum state was generated
- A parameterized quantum model was initialized
- Fidelity between the predicted and target state was computed
- Runtime for fidelity computation was recorded

Each configuration was repeated multiple times to reduce statistical noise.

## Metrics
- Mean fidelity
- Fidelity standard deviation
- Average runtime per evaluation

## Implementation
The experiment was implemented in `scalability.py`.  
Results were aggregated and stored in CSV format for visualization and analysis.

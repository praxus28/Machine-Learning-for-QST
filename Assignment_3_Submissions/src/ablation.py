import numpy as np
from quantum_model import QuantumModel

def ablation_layers(n_qubits=3, layer_list=None, trials=30, seed=1):
    if layer_list is None:
        layer_list = [1, 2, 4, 8]

    rng = np.random.default_rng(seed)
    results = []

    for layers in layer_list:
        fidelities = []
        for _ in range(trials):
            target = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
            target /= np.linalg.norm(target)

            model = QuantumModel(n_qubits, n_layers=layers, seed=rng.integers(1e9))
            fidelities.append(model.fidelity_with(target))

        results.append({
            "layers": layers,
            "mean_fidelity": float(np.mean(fidelities)),
            "std_fidelity": float(np.std(fidelities))
        })

    return results

def summarize_ablation_results(results):
    return results

if __name__ == "__main__":
    results = ablation_layers()
    for r in results:
        print(r)
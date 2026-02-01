import numpy as np
import time
import csv
from quantum_model import QuantumModel
from pathlib import Path


def random_pure_state(dim, rng):
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return vec / np.linalg.norm(vec)


def scalability_experiment(qubit_list, trials=10, n_layers=1, seed=0):
    rng = np.random.default_rng(seed)
    results = []

    for n in qubit_list:
        fidelities = []
        runtimes = []

        for _ in range(trials):
            target = random_pure_state(2 ** n, rng)
            model = QuantumModel(n, n_layers=n_layers, seed=rng.integers(1e9))

            start = time.time()
            fid = model.fidelity_with(target)
            elapsed = time.time() - start

            fidelities.append(fid)
            runtimes.append(elapsed)

        results.append({
            "n_qubits": n,
            "fidelity_mean": float(np.mean(fidelities)),
            "fidelity_std": float(np.std(fidelities)),
            "avg_runtime": float(np.mean(runtimes))
        })

    return results


def save_scalability_summary(results, out_path="outputs/scalability_results.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_qubits",
            "avg_runtime",
            "fidelity_mean",
            "fidelity_std"
        ])
        for r in results:
            writer.writerow([
                r["n_qubits"],
                r["avg_runtime"],
                r["fidelity_mean"],
                r["fidelity_std"]
            ])


if __name__ == "__main__":
    qubits = [1, 2, 3, 4, 5]
    results = scalability_experiment(qubits, trials=20)
    save_scalability_summary(results)
    print("Scalability results saved to outputs/scalability_results.csv")
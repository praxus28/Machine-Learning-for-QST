import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_scalability(csv_path="outputs/scalability_results.csv"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots()

    ax1.errorbar(
        df["n_qubits"],
        df["fidelity_mean"],
        yerr=df["fidelity_std"],
        marker="o",
        label="Fidelity"
    )
    ax1.set_xlabel("Number of qubits")
    ax1.set_ylabel("Mean fidelity")

    ax2 = ax1.twinx()
    ax2.plot(
        df["n_qubits"],
        df["avg_runtime"],
        marker="s",
        color="tab:red",
        label="Runtime"
    )
    ax2.set_ylabel("Runtime (seconds)")

    plt.title("Scalability of tomography surrogate")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_scalability()

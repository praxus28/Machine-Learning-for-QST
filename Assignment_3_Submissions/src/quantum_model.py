import numpy as np
from serialization import save_pickle, load_pickle

class QuantumModel:
    def __init__(self, n_qubits, n_layers=1, params=None, seed=None):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)

        if params is None:
            self.params = self.rng.normal(size=self.dim) + 1j * self.rng.normal(size=self.dim)
        else:
            self.params = params

    def statevector(self):
        vec = self.params.astype(np.complex128)
        norm = np.linalg.norm(vec)
        return vec / norm

    def fidelity_with(self, target_state):
        psi = self.statevector()
        overlap = np.vdot(psi, target_state)
        return np.abs(overlap) ** 2

    def save(self, path):
        payload = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "params": self.params
        }
        save_pickle(payload, path)

    @staticmethod
    def load(path):
        payload = load_pickle(path)
        return QuantumModel(
            payload["n_qubits"],
            payload["n_layers"],
            payload["params"]
        )
    
if __name__ == "__main__":
    model = QuantumModel(n_qubits=2, n_layers=2, seed=42)
    model.save("models/model_trackA_2.pkl")
    print("Model saved to models/model_trackA_2.pkl")
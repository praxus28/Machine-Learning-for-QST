import pickle
from pathlib import Path

def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def demonstrate_serialization_roundtrip():
    test_obj = {
        "n_qubits": 2,
        "params": [1.0, 2.0, 3.0],
        "meta": "roundtrip test"
    }
    path = Path("models/model_test_2.pkl")
    save_pickle(test_obj, path)
    return load_pickle(path)

if __name__ == "__main__":
    obj = demonstrate_serialization_roundtrip()
    print("Serialization roundtrip successful:", obj)
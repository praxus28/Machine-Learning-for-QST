import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class QSTDataset(Dataset):
    def __init__(self, assignment1_root):
        """
        assignment1_root = path to Assignment 1 directory
        """
        meas_dir = Path(assignment1_root) / "data" / "single_qubit" / "measurements"
        recon_dir = Path(assignment1_root) / "data" / "single_qubit" / "reconstructions"

        self.samples = []

        for meas_file in meas_dir.glob("*.npy"):
            name = meas_file.stem
            recon_file = recon_dir / f"{name}_recon.npy"

            meas = np.load(meas_file, allow_pickle=True).item()
            rho = np.load(recon_file)

            # features = expectation values (same as Assignment 1)
            Ex = (meas["X"][0] - meas["X"][1]) / meas["shots"]
            Ey = (meas["Y"][0] - meas["Y"][1]) / meas["shots"]
            Ez = (meas["Z"][0] - meas["Z"][1]) / meas["shots"]

            features = np.array([Ex, Ey, Ez, meas["shots"], 0, 0])

            self.samples.append((features, rho))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, rho = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(rho, dtype=torch.complex64)
        )

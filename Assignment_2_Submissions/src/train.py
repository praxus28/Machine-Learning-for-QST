import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model import DensityMatrixMLP
from dataset import QSTDataset

# ---- paths ----
ASSIGNMENT1_ROOT = Path("../Assignment_1_Submissions")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- dataset ----
dataset = QSTDataset(ASSIGNMENT1_ROOT)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---- model ----
model = DensityMatrixMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Note: Loss may plateau due to small dataset size and limited state diversity.
# ---- training ----
for epoch in range(50):
    total_loss = 0.0
    for x, rho_true in loader:
        rho_pred = model(x)
        loss = torch.mean(torch.abs(rho_pred - rho_true))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss = {total_loss:.6f}")

# ---- save ----
torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
print("Model saved to outputs/model.pt")

import torch
import torch.nn as nn

class DensityMatrixMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # parameters for lower-triangular L
        )

    def forward(self, x):
        params = self.net(x)

        # Lower triangular matrix L
        L = torch.zeros((x.shape[0], 2, 2), dtype=torch.complex64, device=x.device)
        L[:, 0, 0] = torch.exp(params[:, 0])        # positive diagonal
        L[:, 1, 0] = params[:, 1] + 1j * params[:, 2]
        L[:, 1, 1] = torch.exp(params[:, 3])

        rho = L @ torch.conj(L.transpose(-1, -2))
        trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
        rho = rho / trace.unsqueeze(-1).unsqueeze(-1)


        return rho

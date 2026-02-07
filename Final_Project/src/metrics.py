import torch

def fidelity(rho, sigma):
    sqrt_rho = torch.linalg.sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    return torch.real(torch.trace(torch.linalg.sqrtm(product))) ** 2

def trace_distance(rho, sigma):
    diff = rho - sigma
    eigvals = torch.linalg.eigvals(diff)
    return 0.5 * torch.sum(torch.abs(eigvals))
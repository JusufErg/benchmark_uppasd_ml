import torch
import pandas as pd
from .hamiltonian import full_spin_hamiltonian

def optimize_spins(spins_init, J_pairs, DMI_pairs=None, anisotropy_data=None, 
                   lr=0.1, steps=500, simid="default", optimizer_name="adam"):
    """
    Gradient-based spin optimization using PyTorch optimizers (Adam, SGD, LBFGS).
    Logs total and component energies at each step.

    Args:
        spins_init: (N, 3) initial spins
        J_pairs, DMI_pairs, anisotropy_data: interaction data
        optimizer_name: one of "adam", "sgd", "lbfgs"
    Returns:
        Optimized (unit-normalized) spins: (N, 3)
    """
    spins = spins_init.clone().detach().requires_grad_(True)

    # Choose optimizer
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam([spins], lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD([spins], lr=lr)
    elif optimizer_name.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([spins], lr=lr, max_iter=20)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {
        "step": [],
        "total": [],
        "heisenberg": [],
        "dmi": [],
        "anisotropy": []
    }

    for step in range(steps):
        def closure():
            optimizer.zero_grad()
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
            loss = terms["total"]
            loss.backward()
            return loss

        # Run optimizer step
        if optimizer_name.lower() == "lbfgs":
            energy = optimizer.step(closure)
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
        else:
            optimizer.zero_grad()
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
            energy = terms["total"]
            energy.backward()
            optimizer.step()

        # Logging
        history["step"].append(step)
        history["total"].append(terms["total"].item())
        history["heisenberg"].append(terms["heisenberg"].item())
        history["dmi"].append(terms["dmi"].item())
        history["anisotropy"].append(terms["anisotropy"].item())

        if step % 50 == 0:
            print(f"Step {step:03d} | Total: {terms['total'].item():.6f} | "
                  f"Heis: {terms['heisenberg'].item():.6f} | "
                  f"DMI: {terms['dmi'].item():.6f} | "
                  f"Aniso: {terms['anisotropy'].item():.6f}")

    # Save energy log
    df = pd.DataFrame(history)
    df.to_csv(f"data/energy_log_{simid}_{optimizer_name}.csv", index=False)
    print(f"✅ Energy log saved to data/energy_log_{simid}_{optimizer_name}.csv")

    return spins.detach() / spins.detach().norm(dim=1, keepdim=True)

(base) jusufergashev@n158-p91 benchmark_project % nano opt_pytorch/optimize.py 
(base) jusufergashev@n158-p91 benchmark_project % nano opt_pytorch/main.py    
(base) jusufergashev@n158-p91 benchmark_project % cat opt_pytorch/main.py
from opt_pytorch.optimize import optimize_spins
from opt_pytorch.utils import load_spins, load_Jij, load_optional
from config import simid
import time

spins = load_spins(f"data/spins_{simid}.npy")
Jij = load_Jij(f"data/Jij_{simid}.npy")
DMI = load_optional(f"data/DMI_{simid}.npy")
ANISO = load_optional(f"data/anisotropy_{simid}.npy")

for opt_name in ["adam", "sgd", "lbfgs"]:
    print(f"Running optimizer: {opt_name}")
    start =  time.time()
    opt_spins = optimize_spins(
        spins, Jij,
        DMI_pairs=DMI,
        anisotropy_data=ANISO,
        simid=simid,
        optimizer_name=opt_name
)
    duration = time.time() - start
    print(f"Finished {opt_name} in {duration:.2f} seconds")	

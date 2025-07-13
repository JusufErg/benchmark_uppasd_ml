import torch
import pandas as pd
from .hamiltonian import full_spin_hamiltonian

def optimize_spins(spins_init, J_pair, DMI_pairs = None, anisotropy_data = None, lr = 0.1, steps=500, simid="default", optimizer_name="adam"):
  

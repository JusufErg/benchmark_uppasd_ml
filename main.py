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

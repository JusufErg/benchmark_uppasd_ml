# Parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(".."))  # Add project root to Python path
from config import simid, folder_name, base_path

# === Parsers ===

def parse_averages(file_path):
    """Parse averages.simid.out file and return a DataFrame"""
    data = np.loadtxt(file_path)
    df = pd.DataFrame(data, columns=['step', 'Mx', 'My', 'Mz', 'M', 
'std_M'])
    return df

def parse_moments(file_path, n_atoms):
    """Parse moment.simid.out to get spin configuration over time"""
    data = np.loadtxt(file_path)
    n_rows = data.shape[0] # total number of rows
    if n_rows == 0:
        raise ValueError("moment file is empty.")
    n_steps = n_rows // n_atoms # number of time steps (assuming one block per atom per step)
    spins = data[:, :3].reshape((n_steps, n_atoms, 3))
    return spins

def parse_jfile(file_path):
    """Parse jfile and return a list of (i, j, Jij) interactions"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            i, j = int(parts[0]), int(parts[1])
            Jij = float(parts[5])  # in mRy
            data.append((i - 1, j - 1, Jij))  # 0-based indexing
    return data

def parse_dmi(file_path):
    """Parse DMI file and return list of (i, j, Dx, Dy, Dz)"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            i, j = int(parts[0]), int(parts[1])
            Dx, Dy, Dz = map(float, parts[2:5])
            data.append((i - 1, j - 1, Dx, Dy, Dz))  # flat tuple
    return data

def parse_anisotropy(file_path):
    """Parse block-formatted aniso1.*.out file"""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Atom=" in line:
                try:
                    atom_index = int(line.split()[1]) - 1
                    n_vec = list(map(float, lines[i+1].split()))
                    K = float(lines[i+2].split()[0])
                    data.append((atom_index, K, *n_vec))
                    i += 4  # Skip separator line
                except Exception:
                    i += 1  # skip and keep going on error
            else:
                i += 1
    return data

# === Helpers ===

def plot_magnetization(df, title="Average Magnetization vs Time", 
simid="default"):
    plt.figure(figsize=(8, 5))
    plt.plot(df['step'], df['Mx'], label='Mx')
    plt.plot(df['step'], df['My'], label='My')
    plt.plot(df['step'], df['Mz'], label='Mz')
    plt.plot(df['step'], df['M'], label='|M|', linestyle='--')
    plt.xlabel("Time step")
    plt.ylabel("Magnetization")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"../plots/magnetization_{simid}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved plot to {plot_path}")

def normalize_spins(spins):
    """Ensure each spin vector has unit norm"""
    norms = np.linalg.norm(spins, axis=-1, keepdims=True)
    return spins / np.clip(norms, a_min=1e-10, a_max=None)

def save_spins(spins, simid):
    filename = f"../data/spins_{simid}.npy"
    np.save(filename, spins)
    print(f" Saved spins to {filename}")

def save_Jij(data, simid):
    filename = f"../data/Jij_{simid}.npy"
    np.save(filename, data)
    print(f" Saved Jij to {filename}")

# === Main Execution ===

if __name__ == "__main__":

    # Magnetization
    avg_file = os.path.join(base_path, f"averages.{simid}.out")
    if os.path.exists(avg_file):
        print(f" Parsing {avg_file}")
        df = parse_averages(avg_file)
        plot_magnetization(df, simid=simid)
    else:
        print(f" File not found: {avg_file}")

    # Spin configuration
    moment_file = os.path.join(base_path, f"moment.{simid}.out")
    if os.path.exists(moment_file):
        print(f" Parsing {moment_file}")

        moment_file = os.path.join(base_path, f"moment.{simid}.out")
        moment_data = np.loadtxt(moment_file)
        total_rows = moment_data.shape[0]

        # Use anisotropy to infer atom count (works since it's loaded later)
        aniso_path = os.path.join(base_path, f"aniso1.{simid}.out")
        aniso_data = parse_anisotropy(aniso_path)
        n_atoms = len(aniso_data)

        if total_rows % n_atoms != 0:
            raise ValueError("Mismatch between moment file and anisotropy atom count!")

        print(f" Number of atoms detected: {n_atoms}")


        spins = parse_moments(moment_file, n_atoms)
        spins = normalize_spins(spins)
        save_spins(spins, simid)
    else:
        print(f" File not found: {moment_file}")

    # J values
    jfile_path = os.path.join(base_path, "jfile")
    if os.path.exists(jfile_path):
        print(f" Parsing {jfile_path}")
        Jij_data = parse_jfile(jfile_path)
        save_Jij(Jij_data, simid)
        print(f" First 5 J_ij pairs:")
        for i, j, Jij in Jij_data[:5]:
            print(f"   Atom {i} - Atom {j} : J = {Jij} mRy")
    else:
        print(f" J file not found at {jfile_path}")

    # DMI values
    dmi_path = os.path.join(base_path, "dmfile")
    if os.path.exists(dmi_path):
        print(f" Parsing DMI from {dmi_path}")
        dmi_data = parse_dmi(dmi_path)
        np.save(f"../data/DMI_{simid}.npy", dmi_data)
        print(f" Saved DMI to ../data/DMI_{simid}.npy")
    else:
        print(f"️  DMI file not found at {dmi_path}, skipping.")

    # Anisotropy values
    aniso_path = os.path.join(base_path, f"aniso1.{simid}.out")
    if os.path.exists(aniso_path):
        print(f" Parsing anisotropy from {aniso_path}")
        aniso_data = parse_anisotropy(aniso_path)
        np.save(f"../data/anisotropy_{simid}.npy", aniso_data)
        print(f" Saved anisotropy to ../data/anisotropy_{simid}.npy")
    else:
        print(f"️  Anisotropy file not found at {aniso_path}, skipping.")


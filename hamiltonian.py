import torch 


def full_spin_hamiltonian(spins, J_pairs, DMI_pairs=None, 
anisotropy_data=None):
    """
    Compute total energy and return breakdown of Heisenberg, DMI, and 
Anisotropy contributions.

    Returns:
        dict with keys: 'total', 'heisenberg', 'dmi', 'anisotropy'
    """
    e_heis = e_dmi = e_aniso = torch.tensor(0.0, device=spins.device)

    # Heisenberg exchange
    i, j = J_pairs[:, 0].long(), J_pairs[:, 1].long()
    Jij = J_pairs[:, 2]
    e_heis = -torch.sum(Jij * torch.sum(spins[i] * spins[j], dim=1))

    # DMI
    if DMI_pairs is not None:
        i_dmi = DMI_pairs[:, 0].long()
        j_dmi = DMI_pairs[:, 1].long()
        Dij = DMI_pairs[:, 2:5]
        cross = torch.cross(spins[i_dmi], spins[j_dmi], dim=1)
        e_dmi = -torch.sum(torch.sum(Dij * cross, dim=1))

    # Anisotropy
    if anisotropy_data is not None:
        ai = anisotropy_data[:, 0].long()
        Ki = anisotropy_data[:, 1]
        ni = anisotropy_data[:, 2:5]
        proj = torch.sum(spins[ai] * ni, dim=1)
        e_aniso = -torch.sum(Ki * proj ** 2)

    total = e_heis + e_dmi + e_aniso

    return {
        "total": total,
        "heisenberg": e_heis,
        "dmi": e_dmi,
        "anisotropy": e_aniso
    }


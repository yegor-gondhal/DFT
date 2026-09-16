import numpy as np

total_file = np.load("preprocess/total_density.npy", mmap_mode="r")
spin_file = np.load("preprocess/spin_density.npy", mmap_mode="r")
orbital_file = np.load("preprocess/orbital_density.npy", mmap_mode="r")
grid = np.load("preprocess/grid.npy", mmap_mode="r")

with open("data/orbital_data.xyz", "w", buffering=1_000_000) as output_file:
    write = 0
    chunk_size = 1_000_000
    selected_orbital = orbital_file[-1, :]
    size = selected_orbital.shape[0]
    orbital_max = np.max(selected_orbital)
    tau = 1e-5
    orbital_min = tau * orbital_max
    while write != size:
        print(100 * write / size)
        write_to = min(write + chunk_size, size)
        orbital_chunk = selected_orbital[write:write_to]
        orbital_mask = (orbital_chunk > orbital_min)
        orbital_chunk = orbital_chunk[orbital_mask]
        q = np.log10(orbital_chunk)
        q_min = np.log10(orbital_min)
        q_max = np.log10(orbital_max)
        orbital_norm = np.clip((q - q_min) / (q_max - q_min), 0, 1)
        orbital_rgb = np.round(orbital_norm * 255)
        orbital_rgb = np.repeat(orbital_rgb[:, None], 3, axis=1)
        orbital_rgb = orbital_rgb.astype(np.uint8)
        orbital_data = np.column_stack((grid[write:write_to][orbital_mask], orbital_rgb))

        np.savetxt(output_file, orbital_data, fmt=("%.7f", "%.7f", "%.7f", "%d", "%d", "%d"), delimiter=" ")
        write = write_to

with open("data/density_data.xyz", "w", buffering=1_000_000) as output_file:
    write = 0
    chunk_size = 1_000_000
    size = total_file.shape[0]
    density_max = np.max(total_file)
    tau = 1e-5
    density_min = tau * density_max
    while write != size:
        print(100 * write / size)
        write_to = min(write+chunk_size, size)
        dense_chunk = total_file[write:write_to]
        density_mask = (dense_chunk > density_min)
        dense_chunk = dense_chunk[density_mask]
        q = np.log10(dense_chunk)
        q_min = np.log10(density_min)
        q_max = np.log10(density_max)
        density_norm = np.clip((q - q_min) / (q_max - q_min), 0, 1)
        density_rgb = np.round(density_norm * 255)
        density_rgb = np.repeat(density_rgb[:, None], 3, axis=1)
        density_rgb = density_rgb.astype(np.uint8)
        density_data = np.column_stack((grid[write:write_to][density_mask], density_rgb))

        np.savetxt(output_file, density_data, fmt=("%.7f", "%.7f", "%.7f", "%d", "%d", "%d"), delimiter=" ")
        write = write_to


with open("data/spin_data.xyz", "w", buffering=1_000_000) as output_file:
    spin_max = np.max(np.abs(spin_file))
    spin_tau = 1e-5
    spin_min = spin_tau * spin_max
    size = spin_file.shape[0]
    write = 0
    chunk_size = 1_000_000
    while write != size:
        print(100*write/size)
        write_to = min(write+chunk_size, size)
        spin_chunk = spin_file[write:write_to]
        spin_mask = np.abs(spin_chunk) > spin_min
        spin_chunk = spin_chunk[spin_mask]
        spin_norm = 0.5 * (spin_chunk / spin_max + 1)
        spin_rgb = np.round(spin_norm * 255)
        spin_rgb = np.repeat(spin_rgb[:, None], 3, axis=1)
        spin_rgb = spin_rgb.astype(np.uint8)
        spin_data = np.column_stack((grid[write:write_to][spin_mask], spin_rgb))

        np.savetxt(output_file, spin_data, fmt=("%.7f", "%.7f", "%.7f", "%d", "%d", "%d"), delimiter=" ")
        write = write_to


'''
grid /= 5
density_max = xp.max(total_density)
tau = 1e-6
density_min = tau * density_max
density_mask = (total_density > density_min)
total_density = total_density[density_mask]
q = xp.log10(total_density)
q_min = xp.log10(density_min)
q_max = xp.log10(density_max)
density_norm = xp.clip((q - q_min) / (q_max - q_min), 0, 1)
#density_norm = xp.asnumpy(density_norm)

#density_rgba = colormaps["Blues"](density_norm)
#density_rgb = density_rgba[:, :3]
density_rgb = xp.round(density_norm * 255)
density_rgb = xp.repeat(density_rgb[:, None], 3, axis=1)
density_rgb = density_rgb.astype(xp.uint8)
density_data = xp.column_stack((grid[density_mask], density_rgb))

spin_max = xp.max(xp.abs(spin_density))
spin_tau = 1e-4
spin_min = spin_tau * spin_max
spin_mask = (xp.abs(spin_density) > spin_min)
spin_density = spin_density[spin_mask]
spin_norm = 0.5*(spin_density/spin_max + 1)
#spin_norm = xp.asnumpy(spin_norm)
#spin_cmap = colors.LinearSegmentedColormap.from_list("spin_density",["blue", "white", "red"])
#spin_rgb = spin_cmap(spin_norm)
#spin_rgb = spin_rgb[:, :3]
spin_rgb = xp.round(spin_norm * 255)
spin_rgb = xp.repeat(spin_rgb[:, None], 3, axis=1)
spin_rgb = spin_rgb.astype(np.uint8)
spin_data = np.column_stack((grid[spin_mask], spin_rgb))

np.savetxt("data/density_data.xyz", density_data, fmt=["%.7f", "%.7f", "%.7f", "%d", "%d", "%d"], delimiter=" ")
np.savetxt("data/spin_data.xyz", spin_data, fmt=["%.7f", "%.7f", "%.7f", "%d", "%d", "%d"], delimiter=" ")
'''
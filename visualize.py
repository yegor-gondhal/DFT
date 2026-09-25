import numpy as np

total_file = np.load("preprocess/total_density.npy", mmap_mode="r")
spin_file = np.load("preprocess/spin_density.npy", mmap_mode="r")
grid = np.load("preprocess/grid.npy", mmap_mode="r")

print(total_file.shape)

orbitals = np.array([10])

total_file = np.sum(total_file[orbitals], axis=0)
#spin_file = np.sum(spin_file[orbitals], axis=0)

with open(f"data/density3_data.xyz", "w", buffering=1_000_000) as output_file:
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

'''

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
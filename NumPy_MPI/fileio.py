# Parallel decomposition and I/O
from mpi4py import MPI
import h5py

def read_input_files(fname):

    with h5py.File(fname, "r", driver="mpio", comm=comm) as f:
        x = f["x"]
        y = f["y"]
        
        h  = f["h"]
        hu = f["hu"]
        hv = f["hv"]
        z  = f["z"]

        n  = f["n"]
        A_g  = f["A_g"]

        # Global dimensions
        Ny, Nx = dset.shape

        # Compute local domain size
        local_Nx = Nx // dims[1]
        local_Ny = Ny // dims[0]

        # Handle remainder (if Nx or Ny not divisible by dims)
        x0 = px * local_Nx
        y0 = py * local_Ny
        if px == dims[1] - 1:
            local_Nx = Nx - x0
        if py == dims[0] - 1:
            local_Ny = Ny - y0

        # Read local tile (hyperslab)
        local_h = h[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_hu = hu[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_hv = hv[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_z = z[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_A_g = z[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_n = n[y0:y0 + local_Ny, x0:x0 + local_Nx]
        local_x = x[x0:x0 + local_Nx]
        local_y = y[y0:y0 + local_Ny]

    return local_h, local_hu, local_hv, local_z, local_A_g, local_n, local_x, local_y


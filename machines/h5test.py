from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Check if h5py was built with MPI support
print(f"Rank {rank} MPI-enabled h5py = {h5py.get_config().mpi}")

filename = "test_parallel.h5"

# Each rank writes one element
with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
    dset = f.create_dataset("ranks", (size,), dtype='i')
    dset[rank] = rank

# Barrier to ensure all ranks finish writing before reading
comm.Barrier()

# Read back on rank 0
if rank == 0:
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        data = f["ranks"][:]
        print("Data read from file:", data)
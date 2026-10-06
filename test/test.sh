#!/bin/bash -l
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH -t 12:00:00
#SBATCH -o test_suite.log



conda activate mother

module list

export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_io=romio321

# Correctness suite
echo "=== mpi_test.py ==="

srun -n 8 python mpi_test.py

# Edge cases the main suite does not reach
echo "=== mpi_test_mpp_edges.py ==="
srun -n 8 python mpi_test_mpp_edges.py
echo "=== mpi_test_interp_memory.py ==="
srun -n 8 python mpi_test_interp_memory.py
echo "=== test_xnpy_store.py ==="
python test_xnpy_store.py


# Benchmark
echo "=== benchmark.py ==="

srun -n 8 python benchmark.py --size 20000000 --reps 5 --warmup 2


python summarize_benchmarks.py
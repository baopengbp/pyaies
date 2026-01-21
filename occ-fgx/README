                                           Full Grid eXchange(pbc/df/fft)

                       occ-fgx, occ-fgx-pymp, fgx-pymp, fgx-opt, occ-fgx-mpi
1. Install pymp:
pip install pymp-pypi

2. Need more than pyscf input, see the example:
import pbc_df_fft
# occ-fgx pymp
df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_occ_pymp
# occ-fgx
df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_occ 
# fgx pymp
df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_pymp 
# fgx opt using dm2c and incore aokpts
df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_opt
# fgx orig

3.Copy occ-fgx to your_dir and set environment virable, such as:
export PYTHONPATH=$PYTHONPATH:your_dir/occ-fgx 

4. run: OMP_NUM_THREADS=16 python a.py

                    Benchmark

For periodic systems, further optimization of the fully grid-based exchange method for occupied orbitals (occ-fgX) is expected to be theoretically faster by a factor of the basis set number divided by the number of occupied orbitals (N/o) compared to the fully grid-based method (fgX). The method uses full memory to store atomic orbital integrals on the grid, with MPI and PyMP (shared-memory multiprocessing) enabling parallelization by k-points and occupied orbitals in blocks. PyMP significantly reduces memory usage, allows flexible switching between OpenMP multithreading and multiprocessing while maintaining parallelism, and facilitates parallel programming for critical sections of the code. For Diamond PBE0/gth-DZVP with a 2*2*2 k-point mesh, the method achieves a speedup of 6 times, reducing the time to 72 seconds with OpenMP, a speedup of 3 times to 24 seconds with MPI, and PyMP performs 1.6 times faster than MPI, reducing the time to 15 seconds (Table 1). For a 3*3*3 k-point mesh, MPI requires 113 GB of memory, while PyMP only uses 28 GB. 

Table 1.  Diamond DFT Time/SCF cycle(Seconds)
____________________________________
Method	      openMP	MPI	PyMP
____________________________________
occ-fgX	         72	  24	15
fgX	            414 	68	59
Speedup     	  5.7  2.8 4.0
____________________________________
K-points (2,2,2), PBE0/gth-DZVP basis set (104 basis functions), 16 orbitals, using 16-core parallelization.


Table 2.  Diamond HF Time/SCF cycle(Seconds)
____________________________________
Method	      openMP	MPI	 PyMP
____________________________________
occ-fgX	         69	  10  	12
fgX	            426	  54  	56
Speedup 	      6.2  5.4   4.7
____________________________________
K-points (2,2,2), RHF/gth-DZVP basis set (104 basis functions), 16 orbitals, using 16-core parallelization.

5. occ-fgx-mpi version is already in mpi4pyscf dev version, see https://github.com/pyscf/mpi4pyscf/tree/dev/examples/01-parallel_krhf_occ.py and ttps://github.com/pyscf/mpi4pyscf/tree/dev/pbc/df/fft_occk.py    There is a more balenced mpi version in directory mpi using orbitals to partition.
6. Old version is in baopengbp/occ-fft. 

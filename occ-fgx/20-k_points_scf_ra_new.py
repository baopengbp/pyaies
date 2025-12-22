#!/usr/bin/env python

'''
Mean field with k-points sampling

The 2-electron integrals are computed using Poisson solver with FFT by default.
In most scenario, it should be used with pseudo potential.

OMP_NUM_THREADS=1 kernprof -lv 20-k_points_scf_ra.py
'''
import time
from pyscf.pbc import gto, scf, dft, df
import numpy
from pyscf import lib
lib.logger.TIMER_LEVEL = 0
#from pyscf.pbc.dft import multigrid

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-dzvp',
    pseudo = 'gth-pade',
    #rcut = 31.04,
    mesh = [49,49,49]
    #verbose = 4,
)

print('basis=',cell.basis,'nao',cell.nao,'elec',cell.nelec,'precision',cell.precision,cell.mesh,cell.rcut)

Jtime=time.time()
nk = [1,1,2]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
#kmf = scf.KRHF(cell, kpts)
kmf = scf.KRKS(cell, kpts)
kmf.xc = 'pbe0'

import pbc_df_fft
# occ-fgx pymp
#df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_occ_pymp
# occ-fgx
#df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_occ 
# fgx pymp
df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_pymp 
# fgx opt using dm2c and incore aokpts
#df.fft.FFTDF.get_jk = pbc_df_fft.get_jk_opt
# fgx orig

kmf.verbose = 4
kmf.kernel()
print( "Took this long for total: ", time.time()-Jtime)




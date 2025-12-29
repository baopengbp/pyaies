#!/usr/bin/env python

import time
from pyscf import gto, scf, dft, lib
lib.logger.TIMER_LEVEL = 0

mol = gto.M(atom='''C 2.16778 -0.44918 -0.04589
C 1.60190 0.33213 -1.23844
C 0.12647 -0.01717 -1.47395
C -0.77327 0.24355 -0.24682
C 1.32708 -0.12314 1.19439
C -0.15779 -0.45451 0.98457
C 3.64976 -0.11557 0.18351
O -0.82190 -0.04249 2.15412
C -2.23720 -0.18449 -0.53693
C -3.19414 0.02978 0.65049
C -2.83291 0.54586 -1.75716
H -0.78551 -0.74689 2.78026
H 3.79206 0.96776 0.40070
H 4.06333 -0.69236 1.04258
H 4.26160 -0.36608 -0.71348
H -2.98219 -0.66492 1.49356
H -3.14394 1.07829 1.02340
H -4.25233 -0.17395 0.36759
H -2.76237 1.65099 -1.63614
H -2.33431 0.26367 -2.71114
H -3.90769 0.29147 -1.90369
H -2.23894 -1.27795 -0.76804
H -0.76619 1.34209 -0.04759
H -0.29859 -1.55818 0.89954
H 1.70030 1.42806 -1.05330
H 2.19423 0.10028 -2.15602
H -0.21377 0.58584 -2.34721
H 0.04969 -1.09183 -1.76338
H 1.71426 -0.69369 2.07216
H 1.43065 0.96079 1.43691
H 2.08462 -1.54360 -0.25842''',
            charge=-1,
            spin=1,
            basis='ccpvtz'
)
print('basis=',mol.basis,'nao',mol.nao,'elec',mol.nelec)

# occ-DF
#import occdf_override
# optimized df
import dfjk_override
# orig

#mf = scf.ROHF(mol).density_fit()
mf = dft.ROKS(mol).density_fit()
mf.xc = 'b3lyp'
#mf.direct_scf = False
mf.verbose = 4
mf.diis_space = 20
mf.kernel()

grad = mf.nuc_grad_method()
grad.kernel()

exit()

# pyberny
from pyscf.geomopt.berny_solver import optimize
mol_eq = optimize(mf, maxsteps=100)







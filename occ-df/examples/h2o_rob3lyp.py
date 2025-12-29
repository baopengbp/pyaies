#!/usr/bin/env python

import time
from pyscf import gto, scf, dft, lib
lib.logger.TIMER_LEVEL = 0

mol = gto.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
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







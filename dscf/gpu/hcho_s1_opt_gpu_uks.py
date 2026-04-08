#!/usr/bin/env python

import time
from pyscf import gto, scf, dft, lib
lib.logger.TIMER_LEVEL = 0
from pyscf.scf import hf

mol = gto.M(atom='''
C    0.00000000            0.00000000           -0.60298508
O    0.00000000            0.00000000            0.60539399
H    0.00000000            0.93467313           -1.18217476
H    0.00000000           -0.93467313           -1.18217476
''',
            verbose = 4,
            basis='def2-svp')
print('basis=',mol.basis,'nao',mol.nao,'elec',mol.nelec)
homo = mol.nelec[0]-1
l = homo + 1
h = homo 
import cupy as np

en_old = 0

def en_grad(mol):
        global en_old

        mol.spin = 0
        mol.build()
        mf1 = dft.RKS(mol).density_fit().to_gpu()
        mf1.xc = 'pbe0'
        mf1.chkfile = '/tmp/baop/en_g-vipi3o'
        mf1.init_guess = 'chkfile'
        mf1.conv_tol = 1e-11
        mf1.max_cycle = 350
        en0 = mf1.scf()
        if mf1.converged is not True:
            print('g not converge')
            exit()
        mo0 = np.asarray([mf1.mo_coeff, mf1.mo_coeff]).copy()
        mo_occ = np.asarray([mf1.mo_occ/2, mf1.mo_occ/2])

        # s===
        occ = mo_occ.copy()
        occ[0][h]=0      
        occ[0][l]=1
        if not hasattr(mol, 'mo_t'):
          mol.mo_m = mo0
          mol.sm_occ = occ
          mol.mo_t = mo0
        s1 = dft.UKS(mol).density_fit().to_gpu()
        s1.xc = 'pbe0'
        s1 = scf.addons.mom_occ(s1, mo0, occ)

        s1.conv_tol = 1e-11
        s1.max_cycle = 150
        dm_t_init = s1.make_rdm1(mol.mo_m, mol.sm_occ)
        en_mix = s1.scf(dm_t_init)
        mol.mo_m = s1.mo_coeff
        mol.sm_occ = s1.mo_occ
        # grad
        grad = s1.nuc_grad_method()
        grad_mix = grad.kernel()
        # t===
        mol.spin = 2
        mol.build()
        t_occ = mo_occ.copy()
        t_occ[1][h]=0      
        t_occ[0][l]=1
        t1 = dft.UKS(mol).density_fit().to_gpu()
        t1.xc = 'pbe0'
        t1 = scf.addons.mom_occ(t1, mo0, t_occ)

        t1.diis_space = 20
        t1.conv_tol = 1e-11
        t1.max_cycle = 150
        dm_t_init = t1.make_rdm1(mol.mo_t, t_occ)
        en_t = t1.scf(dm_t_init)
        mol.mo_t = t1.mo_coeff
        if en_t > en_mix:
            print('\n Warning: en_t > en_mix \n')
        # grad
        grad = t1.nuc_grad_method()
        grad_t = grad.kernel()
        en = 2*en_mix - en_t
        grad = 2*grad_mix - grad_t
        grad_norm = np.linalg.norm(grad)
        de = en- en_old
        print('tttttt====///////////-----en=', en, 'SEE/eV', (en-en0)*27.2114, 'de=',de, 
              'norm=', grad_norm, 'rms=', grad_norm/np.sqrt(grad.shape[0]), 
              'max grad=', np.max(abs(grad)), '\n',grad)
        en_old = en
        return en, grad
# opt
from pyscf.geomopt import berny_solver, geometric_solver, as_pyscf_method
mf = as_pyscf_method(mol, en_grad)
mol_eq = geometric_solver.optimize(mf, maxsteps=100)





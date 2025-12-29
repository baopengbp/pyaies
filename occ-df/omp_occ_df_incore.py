#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
omp occ incore Density Fitting

Using occ method to make incore DF faster
(ref: Manzer, S.; Horn, P. R.; Mardirossian, N.; Head-Gordon, M. 
J. Chem. Phys. 2015, 143, 024113.)

'MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 python omp_occ_df_incore.py'
'''
import os
import time
import ctypes
import numpy
import scipy.linalg
from scipy.linalg import cho_factor, cho_solve
from pyscf import gto, scf, dft, lib, df, __config__
from pyscf.scf import hf, uhf, chkfile
from pyscf.lib import logger
from pyscf.df import addons
from pyscf.dft import gen_grid, numint
from pyscf.ao2mo import _ao2mo
from pyscf.lo import orth
from functools import reduce

BREAKSYM = getattr(__config__, 'scf_uhf_init_guess_breaksym', True)

def get_jk(mf, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    '''omp occdf version of scf.hf.get_jk function'''
    Jtime=time.time()

    global eri1, c_2c
    mol = mf.mol
    if mf.auxmol is None:    
        print('\n/////   occdf: eri1 compute\n',mf)
        auxmol = df.addons.make_auxmol(mol)
        mf.auxmol = auxmol
# (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        c_2c = cho_factor(int2c)
        eri1 = df.incore.aux_e2(mol, auxmol, 'int3c2e_sph', aosym='s2ij', comp=1).T
        print('auxbasis = ', mf.auxmol.basis)
        print('int2c.shape, eri1.shape', int2c.shape, eri1.shape)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

    if getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
        mo_occ   = numpy.asarray(dm.mo_occ) 
    else:  
        print('init dm has no mo_coeff\n')
        if nset == 1:
            mo_occ, mo_coeff = numpy.linalg.eigh(dm)
        else:
            mo_occa, mo_coeffa = numpy.linalg.eigh(dms[0])
            mo_occb, mo_coeffb = numpy.linalg.eigh(dms[1])
            mo_coeff = numpy.vstack((mo_coeffa, mo_coeffb))
            mo_occ = numpy.vstack((mo_occa, mo_occb))

    nmo = mo_occ.shape[-1]
    mo_coeff = mo_coeff.reshape(-1,nao,nmo)
    mo_occ   = mo_occ.reshape(-1,nmo)

    if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
        mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
        mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
        mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
        assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
        mo_occ = numpy.vstack((mo_occa, mo_occb))
    dmtril = []
    orbo0 = []
    for k in range(nset):
        orbo0.append(numpy.asarray(mo_coeff[k][:,abs(mo_occ[k])>1.E-10], order='F'))
        nocc = orbo0[k].shape[1]
    global iokr, rec
    iokr = []
    rec = []
    naux, nao_pair = eri1.shape
    assert(nao_pair == nao*(nao+1)//2)
    for k in range(nset):
        nocc = orbo0[k].shape[1]
        if nocc > 0:
            buf1 = numpy.empty((naux*nocc,nao))
            fdrv(ftrans, fmmm,
                buf1.ctypes.data_as(ctypes.c_void_p),
                eri1.ctypes.data_as(ctypes.c_void_p),
                orbo0[k].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux), ctypes.c_int(nao),
                (ctypes.c_int*4)(0, nocc, 0, nao),
                null, ctypes.c_int(0))
            buf2r = lib.dot(buf1, orbo0[k]).reshape(naux,nocc,-1)
            buf2r = lib.pack_tril(buf2r)
        iokx = cho_solve(c_2c, buf2r)

        iokx = lib.unpack_tril(iokx.reshape((naux,-1)))
        iokx = numpy.einsum('pij,i->pij', iokx.reshape(naux,nocc,-1), mo_occ[k][abs(mo_occ[k])>1.E-10])
        iokx = iokx.reshape(naux,nocc,-1)

        rec.append(numpy.einsum('kii->k', iokx)) 
        iokr.append(iokx.reshape(naux*nocc,-1)) 
        iokx = None

        kiv = lib.dot(iokr[k].T, buf1) 
        vj[k] = numpy.dot(rec[k].T, eri1)    

        kij = lib.einsum('ui,ju->ij', orbo0[k], kiv)
        kr = scipy.linalg.solve(kij, kiv)
        vk[k] = lib.dot(kiv.T,kr)

    vj = lib.unpack_tril(numpy.asarray(vj), 1).reshape(dm_shape)
    vk = numpy.asarray(vk).reshape(dm_shape)

    print( "Took this long for JK: ", time.time()-Jtime)

    return vj, vk

from pyscf import df
df.df_jk.get_jk = get_jk
print('\n********** occdf jk **********\n')
















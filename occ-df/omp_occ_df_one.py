#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
occ direct Density Fitting

Using occ method to make direct DF faster
(Manzer, S.; Horn, P. R.; Mardirossian, N.; Head-Gordon, M. 
J. Chem. Phys. 2015, 143, 024113.)

openMP: 'MKL_NUM_THREADS=28 OMP_NUM_THREADS=28 python omp_occ_df_direct.py'
'''

import sys
import copy
import time
import ctypes
from functools import reduce
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.outcore import balance_partition
from pyscf import __config__
import numpy
import scipy.linalg
from scipy.linalg import cho_factor, cho_solve
from pyscf import gto, scf, dft, lib, df
from pyscf.df import addons
lib.logger.TIMER_LEVEL = 0

def loop(self, blksize=None):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol
    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)
    int3c='int3c2e'
    int3c = mol._add_suffix(int3c)
    int3c = gto.moleintor.ascint3(int3c)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    naoaux = ao_loc[-1] - nao

    # TODO: Libcint-3.14 and newer version support to compute int3c2e without
    # the opt for the 3rd index.
    #if '3c2e' in int3c:
    #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
    #else:
    #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    #cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)

    comp = 1
    aosym = 's2ij'
    for b0, b1, nL in self.ao_ranges:
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+b1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc)#, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

def get_incore_eri(self, nset, nocca, noccb):
    mol = self.mol
    auxbasis = self.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    self.auxmol = auxmol
# (P|Q)
    int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
    self.c_2c = cho_factor(int2c)
    print('auxbasis = ', self.auxmol.basis, 'int2c.shape', self.c_2c[0].shape)
    naux0 = self.c_2c[0].shape[0]

    int3c='int3c2e'
    int3c = mol._add_suffix(int3c)
    int3c = gto.moleintor.ascint3(int3c)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    naoaux = ao_loc[-1] - nao
    # incore size
    max_memory = (self.max_memory - lib.current_memory()[0])*0.9
    if nset == 1:
        naux_i = int((max_memory*1e6/8 - 2*naux0*nocca*nocca)/(nao*(nao + 1)/2 + nocca*nao))
    else:    
        naux_i = int((max_memory*1e6/8 - 2*naux0*(nocca*nocca + noccb*noccb))/(nao*(nao + 1)/2 + max(nocca, noccb)*nao))
    if naux_i > naoaux:
        naux_i = naoaux
    for i in range(auxmol.nbas+1):
        if ao_loc[mol.nbas+i] >= nao + naoaux - naux_i:
            b0 = i
            break

    # TODO: Libcint-3.14 and newer version support to compute int3c2e without
    # the opt for the 3rd index.
    #if '3c2e' in int3c:
    #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
    #else:
    #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    comp = 1
    aosym = 's2ij'
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+auxmol.nbas)
    buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc, cintopt)
    self.eri1 = numpy.asarray(buf.T, order='C')
    aux_loc = auxmol.ao_loc
    print('incore 3c2e = ',self.eri1.shape, 'memory current use',lib.current_memory()[0])
    # direct size
    max_memory = (self.max_memory - lib.current_memory()[0])*0.7
    if nset == 1:
        blksize = int((max_memory*1e6/8 - naux0*nocca*nocca)/(nao*(nao + 1)/2 + nocca*nao))
    else:    
        blksize = int((max_memory*1e6/8 - naux0*(nocca*nocca + noccb*noccb))/(nao*(nao + 1)/2 + max(nocca, noccb)*nao))
    self.ao_ranges = balance_partition(auxmol.ao_loc, blksize, 0, b0)
    print('direct splits number = ', len(self.ao_ranges))

def get_jk(mf, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mf.stdout, mf.verbose)
    assert(with_j or with_k)

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

    if mf.auxmol is None:    
        print('\n/////   occdf: incore eri1 compute\n',mf)
        if nset == 1:
            nocca = noccb = mo_occ[abs(mo_occ)>1.E-10].shape[0] 
        else:
            nocca = mo_occ[0][abs(mo_occ[0])>1.E-10].shape[0]
            noccb = mo_occ[1][abs(mo_occ[1])>1.E-10].shape[0]
        print('nocca, noccb', nocca, noccb)   
        get_incore_eri(mf, nset, nocca, noccb)
    naux0 = mf.c_2c[0].shape[0]

    if with_k:
#TODO: test whether dm.mo_coeff matching dm
        #mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
        #mo_occ   = numpy.asarray(dm.mo_occ)
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
        obuf2r = []
        kiv = []
        for k in range(nset):
            orbo0.append(numpy.asarray(mo_coeff[k][:,abs(mo_occ[k])>1.E-10], order='F'))
            nocc1 = orbo0[k].shape[1]
            nocc_pair = nocc1*(nocc1+1)//2
            obuf2r.append(numpy.empty((naux0,nocc_pair)))
            kiv.append(numpy.zeros((nocc1,nao)))
        b0 = 0
        for eri1 in loop(mf):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
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
                    obuf2r[k][b0:b1] = lib.pack_tril(buf2r)
            b0 = b1 
            t1 = log.timer_debug1('jk', *t1)
        eri1 = None
        buf1 = None
        buf2r = None
        # incore part
        mf.iokr = []
        mf.rec = []
        naux, nao_pair = mf.eri1.shape
        assert(nao_pair == nao*(nao+1)//2)
        for k in range(nset):
            nocc = orbo0[k].shape[1]
            if nocc > 0:
                buf1 = numpy.empty((naux*nocc,nao))
                fdrv(ftrans, fmmm,
                    buf1.ctypes.data_as(ctypes.c_void_p),
                    mf.eri1.ctypes.data_as(ctypes.c_void_p),
                    orbo0[k].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux), ctypes.c_int(nao),
                    (ctypes.c_int*4)(0, nocc, 0, nao),
                    null, ctypes.c_int(0))
                buf2r = lib.dot(buf1, orbo0[k]).reshape(naux,nocc,-1)
                obuf2r[k][naux0 - naux:] = lib.pack_tril(buf2r)
                buf2r = None
            iokx = cho_solve(mf.c_2c, obuf2r[k])
            iokx = lib.unpack_tril(iokx.reshape((naux0,-1)))
            iokx = numpy.einsum('pij,i->pij', iokx.reshape(naux0,nocc,-1), mo_occ[k][abs(mo_occ[k])>1.E-10])
            iokx = iokx.reshape(naux0,nocc,-1)
            kiv[k] += lib.dot(iokx[naux0 - naux:].reshape(-1,nocc).T, buf1) 
            buf1 = None
            mf.rec.append(numpy.einsum('kii->k', iokx))
            iokx = iokx.reshape(naux0*nocc,-1)
            mf.iokr.append(iokx) 
            iokx = None
            vj[k] += numpy.dot(mf.rec[k][naux0 - naux:].T, mf.eri1)
        # direct part    
        b0 = 0
        c0 = [0] * nset
        c1 = [0] * nset
        for eri1 in loop(mf):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                nocc = orbo0[k].shape[1]
                c1[k] = c0[k] + naux*nocc
                if nocc > 0:
                    buf1 = numpy.empty((naux*nocc,nao))
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo0[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    kiv[k] += lib.dot(mf.iokr[k][c0[k]:c1[k]].T, buf1)
                    if with_j:
                        vj[k] += numpy.dot(mf.rec[k][b0:b1].T, eri1)
                    c0[k] = c1[k]
            b0 = b1         
        for k in range(nset):
# project iv -> uv
            kij = lib.dot(kiv[k], orbo0[k])
            kr = scipy.linalg.solve(kij, kiv[k]) 
            vk[k] = lib.dot(kiv[k].T,kr)
    eri1 = None
    buf1 = None
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    logger.timer(mf, 'vj and vk', *t0)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
df.df_jk.get_jk = get_jk


#!/usr/bin/env python
# occ_DF
# Author: Peng Bao <baopeng@iccas.ac.cn>

# modified from df.df_jk by Peng Bao <baopeng@iccas.ac.cn>

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

    comp = 1
    aosym = 's2ij'
    for b0, b1, nL in self.ao_ranges:
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+b1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc)#, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

def get_incore_eri(self):
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
    max_memory = (self.max_memory - lib.current_memory()[0])*0.8
    naux_i = int((max_memory*1e6/8)/(nao*(nao + 1)/2))
    if naux_i > naoaux:
        naux_i = naoaux
    if naux_i < 10:
        raise ValueError('grids take lots of memory,need more memory or try not save grids ao')
    print(self.max_memory,lib.current_memory()[0],'rrr',ao_loc[mol.nbas+auxmol.nbas],nao,naoaux,naux_i)
    for i in range(auxmol.nbas+1):
        if ao_loc[mol.nbas+i] >= nao + naoaux - naux_i:
            b0 = i
            break
    print('b0 =',b0)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    comp = 1
    aosym = 's2ij'
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+auxmol.nbas)
    buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc, cintopt)
    self.eri1 = numpy.asarray(buf.T, order='C')
    self.eri1_b0 = b0
    aux_loc = auxmol.ao_loc
    print('incore 3c2e = ',self.eri1.shape, 'memory current use',lib.current_memory()[0])
    # direct size
    max_memory = (self.max_memory - lib.current_memory()[0])*0.8
    blksize = int((max_memory*1e6/8)/(nao*(nao + 1)/2))
    self.ao_ranges = balance_partition(auxmol.ao_loc, blksize, 0, b0)
    print('direct splits number = ', len(self.ao_ranges))

def get_jk(mf, dm, hermi=1, vhfopt=None, with_j=True, with_k=True):
    with_j=True
    with_k=False

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mf.stdout, mf.verbose)
    assert(with_j or with_k)

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = numpy.zeros_like(dms)

    if mf.auxmol is None:    
        print('\n/////   df_j: incore eri1 compute\n',mf, '\ndm in dfj is total dm of 2 dimension, so nset=1\n')
        get_incore_eri(mf)
    naux0 = mf.c_2c[0].shape[0]
    if not with_k:
        dmtril = []
        orho = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
                orho.append(numpy.empty((naux0)))
        # direct part for orho
        b0 = 0
        for eri1 in loop(mf):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho = numpy.dot(eri1, dmtril[k])
                    orho[k][b0:b1] = rho
            b0 = b1
        # incore part
        naux, nao_pair = mf.eri1.shape
        assert(nao_pair == nao*(nao+1)//2)
        mf.rec = []
        for k in range(nset):
            if with_j:
                rho = numpy.dot(mf.eri1, dmtril[k])
                orho[k][naux0 - naux:] = rho
            orho[k] = cho_solve(mf.c_2c, orho[k])
            mf.rec.append(orho[k])
            vj[k] += numpy.dot(orho[k][naux0 - naux:].T, mf.eri1)	

        # direct part    
        b0 = 0
        for eri1 in loop(mf):
            naux, nao_pair = eri1.shape
            b1 = b0 + naux
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.dot(orho[k][b0:b1].T, eri1)
            b0 = b1         
    eri1 = None
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = numpy.asarray(vk).reshape(dm_shape)
    logger.timer(mf, 'vj and vk', *t0)
    return vj, vk

def get_j(dfobj, dm, hermi=1, direct_scf_tol=1e-13): 
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    from pyscf.df import addons
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        dfobj.auxmol = auxmol = addons.make_auxmol(mol, dfobj.auxbasis)
        opt = _vhf._VHFOpt(mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
                           dmcondname='CVHFnr_dm_cond',
                           direct_scf_tol=direct_scf_tol)

        # q_cond part 1: the regular int2e (ij|ij) for mol's basis
        opt.init_cvhf_direct(mol, 'int2e', 'CVHFnr_int2e_q_cond')

        # Update q_cond to include the 2e-integrals (auxmol|auxmol)
        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        q_cond = numpy.hstack((opt.q_cond.ravel(), aux_q_cond))
        opt.q_cond = q_cond

        try:
            opt.j2c = j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        dfobj._vjopt = opt

        print('\n/////   df_j: incore eri1 part computed\n')
        get_incore_eri(dfobj)

        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    opt = dfobj._vjopt
    fakemol = opt.fakemol
    dm = numpy.asarray(dm, order='C')
    dm_shape = dm.shape
    nao = dm_shape[-1]
    dm = dm.reshape(-1,nao,nao)
    n_dm = dm.shape[0]
    vj = [0] * n_dm

    # First compute the density in auxiliary basis
    # j3c = fauxe2(mol, auxmol)
    # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
    # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
    nbas = mol.nbas
    nbas1 = mol.nbas + dfobj.auxmol.nbas
    shls_slice1 = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)

    b0 = dfobj.eri1_b0
    nbas1 = mol.nbas + b0
    jaux = numpy.empty((n_dm, opt.j2c[0].shape[0]))
    shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)

    with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
        jaux0 = jk.get_jk(fakemol, dm, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                         aosym='s2ij', hermi=0, shls_slice=shls_slice,
                         vhfopt=opt)
    # remove the index corresponding to bas_placeholder
    naux = opt.j2c[0].shape[0] - dfobj.eri1.shape[0]

    jaux[:,:naux] = numpy.array(jaux0)[:,:,0]
    for k in range(n_dm):
        dmtril = lib.pack_tril(dm[k]+dm[k].T)
        i = numpy.arange(nao)
        dmtril[i*(i+1)//2+i] *= .5
        jaux[k,naux:] = numpy.dot(dfobj.eri1, dmtril)

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)
    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)

    dfobj.rec = []
    dfobj.rec.append(rho.reshape(-1))
    if dfobj.eri1_b0 == 0:
      for k in range(n_dm):
        v0= numpy.dot(rho[naux:,k], dfobj.eri1)#rho
        vj[k] = lib.unpack_tril(v0, 1).reshape((nao,-1))

    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho = rho.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    # TODO: part rho -> vj, 'ijkl,lk->ij' is full dim 
    if dfobj.eri1_b0 != 0:
    # Next compute the Coulomb matrix
    # j3c = fauxe2(mol, auxmol)
    # vj = numpy.einsum('ijk,k->ij', j3c, rho)
    # temporarily set "_dmcondname=None" to skip the call to set_dm method.
      with lib.temporary_env(opt, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                           _dmcondname=None):
        # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
        aux_loc = dfobj.auxmol.ao_loc
        dm_cond = [abs(rho[:,:,i0:i1]).max()
                   for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        opt.dm_cond = numpy.array(dm_cond)
        vj = jk.get_jk(fakemol, rho, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                       aosym='s2ij', hermi=1, shls_slice=shls_slice1,
                       vhfopt=opt)    

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)

    return numpy.asarray(vj).reshape(dm_shape)

# Overwrite the default get_jk to apply the new J/K builder
df.df_jk.get_jk = get_jk
print('<<<<<<<<<<  df_j  >>>>>>>>>>>>>')

from pyscf.dft.numint import NumInt
import num_int
NumInt.nr_rks = num_int.NumInt.nr_rks
NumInt.nr_uks = num_int.NumInt.nr_uks
NumInt.__init__ = num_int.NumInt.__init__
NumInt.block_loop_incore = num_int.NumInt.block_loop_incore

import df_grad_rks
from pyscf.df.grad import rks
rks.Gradients.grad_elec = df_grad_rks.grad_elec
rks.Gradients.get_veff = df_grad_rks.get_veff

import df_grad_uks
from pyscf.df.grad import uks
uks.Gradients.grad_elec = df_grad_uks.grad_elec
uks.Gradients.get_veff = df_grad_uks.get_veff

import df_grad_rhf
from pyscf.df.grad.rhf import Gradients
Gradients.grad_elec = df_grad_rhf.grad_elec

import df_grad_uhf
from pyscf.df.grad.uhf import Gradients
Gradients.grad_elec = df_grad_uhf.grad_elec




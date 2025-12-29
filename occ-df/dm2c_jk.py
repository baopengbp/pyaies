#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

# modified from df.df_jk by Peng Bao <baopeng@iccas.ac.cn>

import time
import numpy
import ctypes
from pyscf import lib
from pyscf.lib import logger
lib.logger.TIMER_LEVEL = 0
from pyscf.ao2mo import _ao2mo
from pyscf import df

def get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert(with_j or with_k)
    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = 0
    vk = numpy.zeros_like(dms)

    if with_j:
        idx = numpy.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

    if not with_k:
        for eri1 in dfobj.loop():
            #rho = numpy.einsum('ix,px->ip', dmtril, eri1)
            #vj += numpy.einsum('ip,px->ix', rho, eri1)
            rho = lib.dot(dmtril, eri1.T)
            vj += lib.dot(rho, eri1)
    else:
        occ_thrd = 1.E-10
        # must direct_scf = False
        if getattr(dm, 'mo_coeff', None) is not None:
            print('dm with mo_coeff')
            mo_coeff = numpy.asarray(dm.mo_coeff, order='F')
            mo_occ   = numpy.asarray(dm.mo_occ)
        else:
            print('dm without mo_coeff')
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
        orbo = []
        for k in range(nset):
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>occ_thrd],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>occ_thrd]))
            orbo.append(numpy.asarray(c, order='F'))

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
        buf = numpy.empty((blksize*nao,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            assert(nao_pair == nao*(nao+1)//2)
            if with_j:
                #rho = numpy.einsum('ix,px->ip', dmtril, eri1)
                #vj += numpy.einsum('ip,px->ix', rho, eri1)
                rho = lib.dot(dmtril, eri1.T)
                vj += lib.dot(rho, eri1)

            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    buf1 = buf[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    #vk[k] += lib.dot(buf1.T, buf1)
                    vk[k] += numpy.dot(buf1.T, buf1)
            t1 = log.timer_debug1('jk', *t1)

    dfobj.iokr = None
    dfobj.rec = None

    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

from pyscf import gto
import scipy.linalg
from pyscf import __config__
MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)  # 2GB
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)

def aux_e2(mol, auxmol, intor='int3c2e', aosym='s1', comp=None, out=None,
           cintopt=None):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Kwargs:
        cintopt : Libcint-3.14 and newer version support to compute int3c2e
            without the opt for the 3rd index.  It can be precomputed to
            reduce the overhead of cintopt initialization repeatedly.

            cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    '''
    from pyscf.gto.moleintor import getints, make_cintopt
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

    # Extract the call of the two lines below
    #  pmol = gto.mole.conc_mol(mol, auxmol)
    #  return pmol.intor(intor, comp, aosym=aosym, shls_slice=shls_slice, out=out)
    intor = mol._add_suffix(intor)
    hermi = 0
    ao_loc = None
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    return getints(intor, atm, bas, env, shls_slice, comp, hermi, aosym,
                   ao_loc, cintopt, out)

def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 #verbose=0, fauxe2=aux_e2):
                 max_memory=MAX_MEMORY, decompose_j2c='cd',
                 lindep=LINEAR_DEP_THR, verbose=0, fauxe2=aux_e2):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    assert(comp == 1)
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    naux = j2c.shape[0]
    log.debug('size of aux basis %d', naux)
    t1 = log.timer('2c2e', *t0)

    j3c = fauxe2(mol, auxmol, intor=int3c, aosym=aosym).reshape(-1,naux)
    t1 = log.timer('3c2e', *t1)

    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        j2c = None
        t1 = log.timer('Cholesky 2c2e', *t1)
        #cderi = scipy.linalg.solve_triangular(low, j3c.T, lower=True,
        #                                      overwrite_b=True)
        cderi = scipy.linalg.blas.dtrsm(
             1, low.T, j3c, side=1, lower=False, overwrite_b=True).T
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        idx = w > LINEAR_DEP_THR
        v = (v[:,idx] / numpy.sqrt(w[idx]))
        cderi = lib.dot(v.T, j3c.T)

    j3c = None
    if cderi.flags.f_contiguous:
        cderi = lib.transpose(cderi.T)
    log.timer('cholesky_eri', *t0)
    return cderi

df.df_jk.get_jk = get_jk
df.incore.cholesky_eri = cholesky_eri
print('\n========= dm2c DF =========\n')






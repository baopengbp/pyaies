#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Modified from pyscf/pbc/df/fft_jk.py
#
# ref: Lin, L. J. Chem. Theory Comput. 2016, 12, 2242\u22122249.
#

'''
JK with occ discrete Fourier transformation
'''

import time
import numpy
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

import scipy
import pymp

from pyscf import __config__

FFT_ENGINE = getattr(__config__, 'pbc_tools_pbc_fft_engine', 'BLAS')

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh

    ni = mydf._numint
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    if hermi == 1 or gamma_point(kpts):
        vR = rhoR = np.zeros((nset,ngrids))
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                rhoR[i,p0:p1] += make_rho(i, ao_ks, mask, 'LDA')
            ao = ao_ks = None

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh).real

    else:  # vR may be complex if the underlying density is complex
        vR = rhoR = np.zeros((nset,ngrids), dtype=np.complex128)
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for i in range(nset):
                for k, ao in enumerate(ao_ks):
                    ao_dm = lib.dot(ao, dms[i,k])
                    rhoR[i,p0:p1] += np.einsum('xi,xi->x', ao_dm, ao.conj())
        rhoR *= 1./nkpts

        for i in range(nset):
            rhoG = tools.fft(rhoR[i], mesh)
            vG = coulG * rhoG
            vR[i] = tools.ifft(vG, mesh)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    weight = cell.vol / ngrids
    vR *= weight
    if gamma_point(kpts_band):
        vj_kpts = np.zeros((nset,nband,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)
    rho = None
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_band):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        for i in range(nset):
            # ni.eval_mat can handle real vR only
            # vj_kpts[i] += ni.eval_mat(cell, ao_ks, 1., None, vR[i,p0:p1], mask, 'LDA')
            for k, ao in enumerate(ao_ks):
                aow = np.einsum('xi,x->xi', ao, vR[i,p0:p1])
                vj_kpts[i,k] += lib.dot(ao.conj().T, aow)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

#occ-fgx
def get_k_kpts_occ(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if mo_coeff is None:
        print('for k point not a*a*a using dm2c')
        mo_coeff = [[]*nset]
        mo_occ = [[]*nset]
        for i in range(nset):
            for k in range(nkpts):
                mocc, c = scipy.linalg.eigh(dms[i][k])
                mo_coeff[i].append(numpy.asarray(c))
                mo_occ[i].append(numpy.asarray(mocc))

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    coords = mydf.grids.coords

    if mydf.first_cyc is None:
        print('Prepare ao2_kpts pymp.shared.array')
        ao2_kpts = pymp.shared.array((nkpts,nao,coords.shape[0]), dtype=numpy.complex128)
        ao2_kpts[:nkpts] = numpy.asarray([numpy.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)])
        mydf.ao2_kpts = ao2_kpts
    ao2_kpts = mydf.ao2_kpts 

    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = numpy.asarray([numpy.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)])
    # occ
    if mydf.first_cyc is None:
            mydf.first_cyc =1
    # occ
    occ_thrd = 1.E-10
    if gamma_point(kpts_band) and gamma_point(kpts):
        for i in range(nset):
            occ = mo_occ[i][0]
            kiv = numpy.zeros((nset,nband,mo_coeff[i][0][:,occ>occ_thrd].shape[1],nao), dtype=dms.dtype)
    else:
        kiv = [[]*nset]
        for i in range(nset):
            for k1 in range(nband):
                occ = mo_occ[i][k1]
                kiv[i].append(numpy.zeros((mo_coeff[i][k1][:,occ>occ_thrd].shape[1],nao), dtype=numpy.complex128))

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'max_memory %s  blksize %d', max_memory, blksize)
    ao1_dtype = numpy.result_type(*ao1_kpts)
    ao2_dtype = numpy.result_type(*ao2_kpts)
    vR_dm = numpy.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    ao_dms_buf = [None] * nkpts
    # occ
    ao3T_buf = [None] * nkpts

    tasks = [(k2,k1,ss) for k2 in range(nkpts) for k1 in range(nband) for ss in range(mo_coeff[0][k1][:,mo_occ[0][k1]>occ_thrd].shape[1])]

    for k2, ao2T in enumerate(ao2_kpts):
        kpt2 = kpts[k2]
        if ao_dms_buf[k2] is None:
                if mo_coeff is None:
                    ao_dms = [lib.dot(dm[k2], ao2T.conj()) for dm in dms]
                else:
                    ao_dms = []
                    for i, dm in enumerate(dms):
                        occ = mo_occ[i][k2]
                        mo_scaled = mo_coeff[i][k2][:,occ>occ_thrd] * numpy.sqrt(occ[occ>occ_thrd])
                        ao_dms.append(lib.dot(mo_scaled.T, ao2T).conj())
                ao_dms_buf[k2] = ao_dms
        else:
                ao_dms = ao_dms_buf[k2]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            if ao2T.size == 0 or ao1T.size == 0:
                continue
        # If we have an ewald exxdiv, we add the G=0 correction near the
        # end of the function to bypass any discretization errors
        # that arise from the FFT.
            mydf.exxdiv = exxdiv
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = numpy.array(1.)
            else:
                expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))

            if ao3T_buf[k1] is None:
                if mo_coeff is not None:           
                    ao3T = []
                    for i, dm in enumerate(dms):
                        occ = mo_occ[i][k1]
                        mo_scaled = mo_coeff[i][k1][:,occ>occ_thrd] 
                        ao3T.append(lib.dot(mo_scaled.T, ao1T))
                ao3T_buf[k1] = ao3T
            else:
                ao3T = ao3T_buf[k1]

            for i in range(nset):
                m3T = ao3T[i].shape[0]
                for p0, p1 in lib.prange(0, m3T, blksize):
                    #print(p.thread_num,'hh',p0, p1,tt0, tt1, blksize)
                    rho1 = numpy.einsum('ig,jg->ijg',
                                        ao3T[i][p0:p1].conj()*expmikr,
                                        ao_dms[i].conj())
                    vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                    rho1 = None
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(p1-p0,-1,ngrids)
                    vG = None
                    if vR_dm.dtype == numpy.double:
                        vR = vR.real
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                tt = mo_coeff[i][k1][:,occ>occ_thrd].shape[1]
                kiv[i][k1] += weight * lib.dot(vR_dm[i, 0:tt], ao1T.T)

    for i in range(nset):
            for k1 in range(nband):
                kij = lib.einsum('ui,ju->ij', mo_coeff[i][k1][:,mo_occ[i][k1]>occ_thrd], kiv[i][k1])
                kr = scipy.linalg.solve(kij.conj(), kiv[i][k1])
                vk_kpts[i,k1] = lib.dot(kiv[i][k1].T.conj(),kr)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real

    if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

# occ-fgx-pymp
def get_k_kpts_occ_pymp(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        #first_cyc = 1
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if mo_coeff is None:
        if mydf.first_cyc is None:
            print('for k point not a*a*a using dm2c')
        mo_coeff = [[]*nset]
        mo_occ = [[]*nset]
        for i in range(nset):
            for k in range(nkpts):
                mocc, c = scipy.linalg.eigh(dms[i][k])
                mo_coeff[i].append(numpy.asarray(c))
                mo_occ[i].append(numpy.asarray(mocc))

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    coords = mydf.grids.coords

    if mydf.first_cyc is None:
        print('Prepare ao2_kpts pymp.shared.array')
        ao2_kpts = pymp.shared.array((nkpts,nao,coords.shape[0]), dtype=numpy.complex128)
        ao2_kpts[:nkpts] = numpy.asarray([numpy.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)])
        mydf.ao2_kpts = ao2_kpts
    ao2_kpts = mydf.ao2_kpts 

    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = numpy.asarray([numpy.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)])
    # occ
    if mydf.first_cyc is None:
            mydf.first_cyc =1
    # occ
    occ_thrd = 1.E-10
    if gamma_point(kpts_band) and gamma_point(kpts):
        for i in range(nset):
            occ = mo_occ[i][0]
            kiv = numpy.zeros((nset,nband,mo_coeff[i][0][:,occ>occ_thrd].shape[1],nao), dtype=dms.dtype)
    else:
        kiv = [[]*nset]
        for i in range(nset):
            for k1 in range(nband):
                occ = mo_occ[i][k1]
                kiv[i].append(numpy.zeros((mo_coeff[i][k1][:,occ>occ_thrd].shape[1],nao), dtype=numpy.complex128))

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'max_memory %s  blksize %d', max_memory, blksize)
    ao1_dtype = numpy.result_type(*ao1_kpts)
    ao2_dtype = numpy.result_type(*ao2_kpts)
    vR_dm = numpy.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    ao_dms_buf = [None] * nkpts
    # occ
    ao3T_buf = [None] * nkpts

    tasks = [(k2,k1,ss) for k2 in range(nkpts) for k1 in range(nband) for ss in range(mo_coeff[0][k1][:,mo_occ[0][k1]>occ_thrd].shape[1])]

    numth = lib.num_threads()
    # lib.num_threads must be 1
    lib.num_threads(1)
    tlist = pymp.shared.list()
    with pymp.Parallel(numth) as p: 
      split1 = np.array_split(tasks,p.num_threads,axis = 0)
      tk = split1[p.thread_num]
      tk2_0 = tk[0][0]
      tk2_1 = tk[-1][0]+1
      for k2 in range(tk2_0,tk2_1):
        tk2 = []
        for i in range(tk.shape[0]):
            if tk[i][0] == k2:
                tk2.append(tk[i]) 
        atk2 =  lib.asarray(tk2)              
        tk1_0 = atk2[0][1]
        tk1_1 = atk2[-1][1]+1
        for k1 in range(tk1_0,tk1_1):
            aaa=0
            tk1 = []
            for i in range(atk2.shape[0]):
                if atk2[i][1] == k1:
                    tk1.append(atk2[i])
            atk1 =  lib.asarray(tk1)              
            tt0 = atk1[0][2]
            tt1 = atk1[-1][2]+1

            ao1T = ao1_kpts[k1]
            ao2T = ao2_kpts[k2]
            kpt1 = kpts_band[k1]
            kpt2 = kpts[k2]
            if ao2T.size == 0 or ao1T.size == 0:
                continue
        # If we have an ewald exxdiv, we add the G=0 correction near the
        # end of the function to bypass any discretization errors
        # that arise from the FFT.
            mydf.exxdiv = exxdiv
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = numpy.array(1.)
            else:
                expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))

            if ao_dms_buf[k2] is None:
                if mo_coeff is None:
                    ao_dms = [lib.dot(dm[k2], ao2T.conj()) for dm in dms]
                else:
                    ao_dms = []
                    for i, dm in enumerate(dms):
                        occ = mo_occ[i][k2]
                        mo_scaled = mo_coeff[i][k2][:,occ>occ_thrd] * numpy.sqrt(occ[occ>occ_thrd])
                        ao_dms.append(lib.dot(mo_scaled.T, ao2T).conj())
                ao_dms_buf[k2] = ao_dms
            else:
                ao_dms = ao_dms_buf[k2]

            if ao3T_buf[k1] is None:
                if mo_coeff is not None:           
                    ao3T = []
                    for i, dm in enumerate(dms):
                        occ = mo_occ[i][k1]
                        mo_scaled = mo_coeff[i][k1][:,occ>occ_thrd] 
                        ao3T.append(lib.dot(mo_scaled.T, ao1T))
                ao3T_buf[k1] = ao3T
            else:
                ao3T = ao3T_buf[k1]

            for i in range(nset):
                for p0, p1 in lib.prange(tt0, tt1, blksize):
                    rho1 = numpy.einsum('ig,jg->ijg',
                                        ao3T[i][p0:p1].conj()*expmikr,
                                        ao_dms[i].conj())
                    vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                    rho1 = None
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(p1-p0,-1,ngrids)
                    vG = None
                    if vR_dm.dtype == numpy.double:
                        vR = vR.real
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                kiv[i][k1][tt0:tt1] += weight * lib.dot(vR_dm[i,tt0:tt1], ao1T.T)
      for _ in p.range(numth):
        tlist.append(kiv)
    lib.num_threads(numth)
    kiv_sh = numpy.sum(numpy.asarray(tlist), axis=0)

    for i in range(nset):
            for k1 in range(nband):
                kij = lib.einsum('ui,ju->ij', mo_coeff[i][k1][:,mo_occ[i][k1]>occ_thrd], kiv_sh[i][k1])
                kr = scipy.linalg.solve(kij.conj(), kiv_sh[i][k1])
                vk_kpts[i,k1] = lib.dot(kiv_sh[i][k1].T.conj(),kr)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real

    if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

# fgx-pymp
def get_k_kpts_pymp(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None, first_cyc=None):
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if mo_coeff is None:
        print('for k point not a*a*a using dm2c')
        mo_coeff = [[]*nset]
        mo_occ = [[]*nset]
        for i in range(nset):
            for k in range(nkpts):
                mocc, c = scipy.linalg.eigh(dms[i][k])
                mo_coeff[i].append(numpy.asarray(c))
                mo_occ[i].append(numpy.asarray(mocc))

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    coords = mydf.grids.coords

    if mydf.first_cyc is None:
        print('Prepare ao2_kpts pymp.shared.array')
        ao2_kpts = pymp.shared.array((nkpts,nao,coords.shape[0]), dtype=numpy.complex128)
        ao2_kpts[:nkpts] = numpy.asarray([numpy.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)])
        mydf.ao2_kpts = ao2_kpts
    ao2_kpts = mydf.ao2_kpts 

    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = numpy.asarray([numpy.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)])
    # for aokpts incore
    if mydf.first_cyc is None:
            mydf.first_cyc =1

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'max_memory %s  blksize %d', max_memory, blksize)
    ao1_dtype = numpy.result_type(*ao1_kpts)
    ao2_dtype = numpy.result_type(*ao2_kpts)
    vR_dm = numpy.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    ao_dms_buf = [None] * nkpts
    # occ
    ao3T_buf = [None] * nkpts

    tasks = [(k2,k1,ss) for k2 in range(nkpts) for k1 in range(nband) for ss in range(nao)]

    numth = lib.num_threads()
    # lib.num_threads must be 1
    lib.num_threads(1)
    tlist = pymp.shared.list()
    with pymp.Parallel(numth) as p: 
     split1 = np.array_split(tasks,p.num_threads,axis = 0)
     tk = split1[p.thread_num]
     tk2_0 = tk[0][0]
     tk2_1 = tk[-1][0]+1
     for k2 in range(tk2_0,tk2_1):
       tk2 = []
       for i in range(tk.shape[0]):
           if tk[i][0] == k2:
               tk2.append(tk[i]) 
       atk2 =  lib.asarray(tk2)              
       tk1_0 = atk2[0][1]
       tk1_1 = atk2[-1][1]+1
       for k1 in range(tk1_0,tk1_1):
        aaa=0
        tk1 = []
        for i in range(atk2.shape[0]):
            if atk2[i][1] == k1:
                tk1.append(atk2[i])
        atk1 =  lib.asarray(tk1)              
        tt0 = atk1[0][2]
        tt1 = atk1[-1][2]+1

        ao1T = ao1_kpts[k1]
        ao2T = ao2_kpts[k2]
        kpt1 = kpts_band[k1]
        kpt2 = kpts[k2]
        if ao2T.size == 0 or ao1T.size == 0:
            continue

        # If we have an ewald exxdiv, we add the G=0 correction near the
        # end of the function to bypass any discretization errors
        # that arise from the FFT.
        mydf.exxdiv = exxdiv
        if exxdiv == 'ewald' or exxdiv is None:
            coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
        else:
            coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
        if is_zero(kpt1-kpt2):
            expmikr = numpy.array(1.)
        else:
            expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))

        if ao_dms_buf[k2] is None:
            if mo_coeff is None:
                ao_dms = [lib.dot(dm[k2], ao2T.conj()) for dm in dms]
            else:
                ao_dms = []
                for i, dm in enumerate(dms):
                    occ = mo_occ[i][k2]
                    mo_scaled = mo_coeff[i][k2][:,occ>0] * numpy.sqrt(occ[occ>0])
                    ao_dms.append(lib.dot(mo_scaled.T, ao2T).conj())
            ao_dms_buf[k2] = ao_dms
        else:
            ao_dms = ao_dms_buf[k2]

        for p0, p1 in lib.prange(tt0, tt1, blksize):
                for i in range(nset):
                    rho1 = numpy.einsum('ig,jg->ijg',
                                        ao1T[p0:p1].conj()*expmikr,
                                        ao_dms[i].conj())
                    vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                    rho1 = None
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(p1-p0,-1,ngrids)
                    vG = None
                    if vR_dm.dtype == numpy.double:
                        vR = vR.real
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR = None
        vR_dm *= expmikr.conj()

        for i in range(nset):
            vk_kpts[i,k1,tt0:tt1] += weight * lib.dot(vR_dm[i,tt0:tt1], ao1T.T)
     for _ in p.range(numth):
         tlist.append(vk_kpts)
    lib.num_threads(numth)
    vk_kpts = numpy.sum(numpy.asarray(tlist), axis=0)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real

    if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def get_k_kpts_opt(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None, first_cyc=None):
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    if mo_coeff is None:
        print('for k point not a*a*a using dm2c')
        mo_coeff = [[]*nset]
        mo_occ = [[]*nset]
        for i in range(nset):
            for k in range(nkpts):
                mocc, c = scipy.linalg.eigh(dms[i][k])
                mo_coeff[i].append(numpy.asarray(c))
                mo_occ[i].append(numpy.asarray(mocc))

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    coords = mydf.grids.coords

    if mydf.first_cyc is None:
        print('Prepare ao2_kpts pymp.shared.array')
        ao2_kpts = pymp.shared.array((nkpts,nao,coords.shape[0]), dtype=numpy.complex128)
        ao2_kpts[:nkpts] = numpy.asarray([numpy.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)])
        mydf.ao2_kpts = ao2_kpts
    ao2_kpts = mydf.ao2_kpts

    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = numpy.asarray([numpy.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)])
    if mydf.first_cyc is None:
            mydf.first_cyc =1

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'max_memory %s  blksize %d', max_memory, blksize)
    ao1_dtype = numpy.result_type(*ao1_kpts)
    ao2_dtype = numpy.result_type(*ao2_kpts)
    vR_dm = numpy.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    ao_dms_buf = [None] * nkpts
    # occ
    ao3T_buf = [None] * nkpts

    tasks = [(k1,k2) for k2 in range(nkpts) for k1 in range(nband)]

    for k1, k2 in tasks:
        ao1T = ao1_kpts[k1]
        ao2T = ao2_kpts[k2]
        kpt1 = kpts_band[k1]
        kpt2 = kpts[k2]
        if ao2T.size == 0 or ao1T.size == 0:
            continue

        # If we have an ewald exxdiv, we add the G=0 correction near the
        # end of the function to bypass any discretization errors
        # that arise from the FFT.
        mydf.exxdiv = exxdiv
        if exxdiv == 'ewald' or exxdiv is None:
            coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
        else:
            coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
        if is_zero(kpt1-kpt2):
            expmikr = numpy.array(1.)
        else:
            expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))

        if ao_dms_buf[k2] is None:
            if mo_coeff is None:
                ao_dms = [lib.dot(dm[k2], ao2T.conj()) for dm in dms]
            else:
                ao_dms = []
                for i, dm in enumerate(dms):
                    occ = mo_occ[i][k2]
                    mo_scaled = mo_coeff[i][k2][:,occ>0] * numpy.sqrt(occ[occ>0])
                    ao_dms.append(lib.dot(mo_scaled.T, ao2T).conj())
            ao_dms_buf[k2] = ao_dms
        else:
            ao_dms = ao_dms_buf[k2]

        if mo_coeff is None:
            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = numpy.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,nao,ngrids)
                vG = None
                if vR_dm.dtype == numpy.double:
                    vR = vR.real
                for i in range(nset):
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
        else:
            for p0, p1 in lib.prange(0, nao, blksize):
                for i in range(nset):
                    rho1 = numpy.einsum('ig,jg->ijg',
                                        ao1T[p0:p1].conj()*expmikr,
                                        ao_dms[i].conj())
                    vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                    rho1 = None
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(p1-p0,-1,ngrids)
                    vG = None
                    if vR_dm.dtype == numpy.double:
                        vR = vR.real
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR = None
        vR_dm *= expmikr.conj()

        for i in range(nset):
            vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real

    if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

from pyscf.lib import logger
def get_k_kpts_omp_orig(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]

    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                  max_memory, blksize)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = (logger.process_clock(), logger.perf_counter())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
        t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)







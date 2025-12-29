#!/usr/bin/env python
# incore DF-gradients
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
Non-relativistic unrestricted DF analytical nuclear gradients
(ref: Bostrom, J.; Aquilante, F.; Pedersen, T. B.; Lindh, R. J. Chem. Theory Comput. 2012, 9, 204.)
'''

from pyscf.grad import rhf as rhf_grad
import time
import ctypes
from pyscf.lib import logger
import numpy
import scipy.linalg
from pyscf import gto, scf, dft, lib
from pyscf.df import addons
from pyscf import df
from pyscf.scf import uhf
from pyscf.ao2mo import _ao2mo
from scipy.linalg import cho_factor, cho_solve

def loop(self, blksize=None):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol #= addons.make_auxmol(self.mol, self.auxbasis)
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
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)

    if blksize is None:
        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
        blksize = min(int(max_memory*1e6/48/nao/(nao+1)), 80)
    comp = 1
    aosym = 's2ij'
    for b0, b1 in self.with_df.prange(0, auxmol.nbas, blksize):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+b0, mol.nbas+b1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                      aosym, ao_loc, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

def get_iokr(mf,  mo_coeff, mo_occ, nset):
    self = mf
    mol = self.mol
    auxbasis = self.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    self.auxmol = auxmol
# (P|Q)
    int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
    c_2c = cho_factor(int2c)
    naux0 = c_2c[0].shape[0]
    nao = mo_coeff.shape[-2]
    print('auxbasis = ', self.auxmol.basis, 'int2c.shape', naux0, nao)

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    if 0==0:
        dmtril = []
        orbo0 = []
        obuf2r = []
        for k in range(nset):
            orbo0.append(numpy.asarray(mo_coeff[k][:,abs(mo_occ[k])>1.E-10], order='F'))
            nocc1 = orbo0[k].shape[1]
            nocc_pair = nocc1*(nocc1+1)//2
            obuf2r.append(numpy.empty((naux0,nocc_pair)))
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

        rec = []
        iokr = []
        for k in range(nset):
            nocc = orbo0[k].shape[1]
            iokx = cho_solve(c_2c, obuf2r[k])
            iokx = lib.unpack_tril(iokx.reshape((naux0,-1)))
            iokx = numpy.einsum('pij,i->pij', iokx.reshape(naux0,nocc,-1), mo_occ[k][abs(mo_occ[k])>1.E-10])
            iokx = iokx.reshape(naux0,nocc,-1)
            buf1 = None
            rec.append(numpy.einsum('kii->k', iokx))
            iokx = iokx.reshape(naux0*nocc,-1)
            iokr.append(iokx) 
            iokx = None   

    return iokr, rec

def loop_aux2(self, intor='int3c2e_ip1', aosym='s1', comp=3):
    mol = self.mol
    auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
    int3c = 'int3c2e_ip1'
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
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)

    int3c2 = 'int3c2e_ip2'
    int3c2 = mol._add_suffix(int3c2)
    int3c2 = gto.moleintor.ascint3(int3c2)
    cintopt2 = gto.moleintor.make_cintopt(atm, bas, env, int3c2)
    ao_loc2 = gto.moleintor.make_loc(bas, int3c2)

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    auxbasis = self.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    aux_offset = auxmol.offset_nr_by_atom()

    for j, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shl0a, shl1a, aux0, aux1 = aux_offset[ia]

        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+shl0a, mol.nbas+shl1a)
        buf1 = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
        shls_slice2 = (0, mol.nbas, 0, mol.nbas, mol.nbas+shl0a, mol.nbas+shl1a)
        buf2 = gto.moleintor.getints3c(int3c2, atm, bas, env, shls_slice2, comp,
                                  's2ij', ao_loc2, cintopt2)
        yield j, aux0, aux1, buf1, buf2

def get_jkgrd(mf, dm, mo_coeff=None, mo_occ=None):
    mol = mf.mol
    auxbasis = mf.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    print('occdf u_grad', 'number of AOs', nao)
    print('number of auxiliary basis functions', naux)
    int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]

    nmo = mo_occ.shape[-1]
    mo_coeff = mo_coeff.reshape(-1,nao,nmo)
    mo_occ   = mo_occ.reshape(-1,nmo)
    if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
        mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
        mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
        mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
        assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
        mo_occ = numpy.vstack((mo_occa, mo_occb))

    orbo = []
    nocc = []
    for k in range(nset):
        c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                         numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
        orbo.append(numpy.asarray(c, order='F'))
        nocc.append(orbo[k].shape[1])

    if mf.with_df.iokr is not None: 
        iokr = mf.with_df.iokr
        rec = mf.with_df.rec
    else:
        iokr, rec = get_iokr(mf, mo_coeff, mo_occ, nset) 

    dmtril = []
    for k in range(nset):
        dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
        i = numpy.arange(nao)
        dmtril[k][i*(i+1)//2+i] *= .5
    dmtrilt = dmtril[0] + dmtril[1]   

    ej = numpy.zeros((mol.natm,3))
    ek = numpy.zeros((mol.natm,3))

    dmt = dms[0]+dms[1]
    rect = rec[0]+rec[1]

    coeff3mo = []
    x3mo = []
    n3mo = []
    ec1_3cu_v = []
    ex1_3cu_v = []
    ex1_3cp = [0]*nset
    ex1_2c = [0]*nset
    for k in range(nset):
        coeff3mo.append(iokr[k].reshape(-1,nocc[k],nocc[k]))
        x3mo.append(numpy.tensordot(coeff3mo[k], coeff3mo[k], axes=([1,2],[1,2])))
        n3mo.append(numpy.tensordot(coeff3mo[k], orbo[k], axes=([2],[1]))) #pon
        ec1_3cu_v.append(numpy.zeros((3,nao,nocc[k]))) 
        ex1_3cu_v.append(numpy.zeros((3,nao)))

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1 # asymmetric
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1
    null = lib.c_null_ptr()

    fmmm_e2 = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv_e2 = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans_e2 = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null_e2 = lib.c_null_ptr() 

    for j, aux0, aux1, int3c_e1, int3c_e2 in loop_aux2(mf, intor='int3c2e_ip1', aosym='s1', comp=3):

        ec1_3cp = numpy.dot(numpy.dot(int3c_e2.swapaxes(1,2), dmtrilt),rect[aux0:aux1])

        tmp = int2c_e1[:,aux0:aux1,:].dot(rect)    
        ec1_2c = numpy.dot(tmp,rect[aux0:aux1])

        for k in range(nset):
            tmp = numpy.empty((3, aux1-aux0, nocc[k], nao))
            fdrv_e2(ftrans_e2, fmmm_e2,
                tmp.ctypes.data_as(ctypes.c_void_p),
                int3c_e2.swapaxes(1,2).ctypes.data_as(ctypes.c_void_p),
                orbo[k].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(3*(aux1-aux0)), ctypes.c_int(nao),
                (ctypes.c_int*4)(0, nocc[k], 0, nao),
                null_e2, ctypes.c_int(0)) #xpon
            ex1_3cp[k] = numpy.tensordot(tmp, n3mo[k][aux0:aux1], axes=([1,2,3],[0,1,2]))

            fdrv(ftrans, fmmm, 
                tmp.ctypes.data_as(ctypes.c_void_p),
                int3c_e1.transpose (0,3,2,1).ctypes.data_as(ctypes.c_void_p),
                orbo[k].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int (3*(aux1-aux0)), ctypes.c_int (nao),
                (ctypes.c_int*4)(0, nocc[k], 0, nao),
                null, ctypes.c_int(0))
            ex1_3cu_v[k] += lib.einsum('xpon,pon->xn', tmp, n3mo[k][aux0:aux1])

            for i in range(3):
                ec1_3cu_v[k][i] += numpy.dot (rect[aux0:aux1], tmp[i].reshape (aux1-aux0, -1)).reshape (nocc[k], nao).T

            ex1_2c[k] = numpy.einsum('xpq,pq->x', int2c_e1[:,aux0:aux1,:], x3mo[k][aux0:aux1])

        ej[j] = -ec1_3cp + ec1_2c 
        for k in range(nset):  
            ek[j] += ex1_3cp[k] - ex1_2c[k]

    aoslices = mol.aoslice_by_atom()
    for j, ia in enumerate(range(mol.natm)):
        shl0, shl1, p0, p1 = aoslices[ia]
        for k in range(nset):
            ej[j] +=  -2 * numpy.tensordot(ec1_3cu_v[k][:,p0:p1], orbo[k][p0:p1], axes=([1,2],[0,1]))
            ek[j] +=  2 * numpy.einsum('xi->x', ex1_3cu_v[k][:,p0:p1])
    return ej, ek

def get_jgrd(mf, dm, mo_coeff=None, mo_occ=None):
    mol = mf.mol
    auxbasis = mf.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    print('dfj u_grad', 'number of AOs', nao)
    print('number of auxiliary basis functions', naux)
    int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

    nmo = mo_occ.shape[-1]
    mo_coeff = mo_coeff.reshape(-1,nao,nmo)
    mo_occ   = mo_occ.reshape(-1,nmo)
    if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
        mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
        mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
        mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
        assert(mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
        mo_occ = numpy.vstack((mo_occa, mo_occb))

    orbo = []
    for k in range(nset):
        c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                         numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
        orbo.append(numpy.asarray(c, order='F'))
    # from df_j.py
    rec = mf.with_df.rec

    dmtril = []
    for k in range(nset):
        dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
        i = numpy.arange(nao)
        dmtril[k][i*(i+1)//2+i] *= .5
    dmtrilt = dmtril[0] + dmtril[1] 
    dmt = dms[0]+dms[1]
    rect = rec[0]

    ej = numpy.empty((mol.natm,3))
    ek = numpy.zeros((mol.natm,3))

    ec1_3cu_v = numpy.zeros((3,nao,nao)) 

    for j, aux0, aux1, int3c_e1, int3c_e2 in loop_aux2(mf, intor='int3c2e_ip1', aosym='s1', comp=3):
        ec1_3cp = numpy.dot(numpy.dot(int3c_e2.swapaxes(1,2), dmtrilt),rect[aux0:aux1])

        tmp = int2c_e1[:,aux0:aux1,:].dot(rect)    
        ec1_2c = numpy.dot(tmp,rect[aux0:aux1])
        for i in range(3):
            tmp = numpy.dot(rect[aux0:aux1], int3c_e1[i].T.reshape (aux1-aux0, -1))
            ec1_3cu_v[i] += tmp.reshape(nao,-1)
        ej[j] = -ec1_3cp + ec1_2c 

    aoslices = mol.aoslice_by_atom()
    for l, ia in enumerate(range(mol.natm)):
        shl0, shl1, p0, p1 = aoslices[ia]
        ej[l] +=  -2 * numpy.tensordot(ec1_3cu_v[:,:,p0:p1], dmt[p0:p1], axes=([2,1],[0,1]))
    return ej, ek

def rhf_make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mo0 = mo_coeff[:,mo_occ>0]
    mo0e = mo0 * (mo_energy[mo_occ>0] * mo_occ[mo_occ>0])
    return numpy.dot(mo0e, mo0.T.conj())

def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    return numpy.asarray((rhf_make_rdm1e(mo_energy, mo_coeff, mo_occ[0]),
                          rhf_make_rdm1e(mo_energy, mo_coeff, mo_occ[1])))

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    #vhf = mf_grad.get_veff(mol, dm0)
    ej, ek = get_jkgrd(mf, dm0, mo_coeff, mo_occ)
    #print('ddd',ej,ek,mo_occ)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)#
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0_sf)
        de[k] += ej[k] + ek[k]
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de





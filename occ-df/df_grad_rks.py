#!/usr/bin/env python
# incore DF-gradients
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
Non-relativistic restricted DF-KS analytical nuclear gradients
(ref: Bostrom, J.; Aquilante, F.; Pedersen, T. B.; Lindh, R. J. Chem. Theory Comput. 2012, 9, 204.)
'''

from pyscf.grad import rhf as rhf_grad
import time
import ctypes
from pyscf.lib import logger
import numpy
import scipy.linalg
from pyscf import gto, scf, dft, lib, grad
from pyscf.df import addons
from pyscf import df
from pyscf.scf import uhf
from pyscf.dft import numint, gen_grid
from pyscf.grad import rks as rks_grad

import df_grad_rhf

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')

    #enabling range-separated hybrids
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    print('hyb',hyb)

    vhf = mf_grad.get_veff(mol, dm0)

    if abs(hyb) < 1e-10:
        ej, ek = df_grad_rhf.get_jgrd(mf, dm0, mo_coeff, mo_occ)
    else:
        ej, ek = df_grad_rhf.get_jkgrd(mf, dm0, mo_coeff, mo_occ)

    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()

    auxbasis = mf.auxbasis
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    aux_offset = auxmol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        aux0, aux1 = aux_offset[ia][2:]
#        print('atom %d %s, shell range %s:%s, AO range %s:%s, aux-AO range %s:%s' %
#             (ia, mol.atom_symbol(ia), shl0, shl1, p0, p1, aux0, aux1))
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0)
# nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
        de[k] += numpy.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
        # + jk
        de[k] += ej[k] + hyb * ek[k]

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        if mf_grad.grid_response: # Only effective in DFT gradients
            print('qq',k,de[k],vhf.exc1_grid[ia])
            de[k] += vhf.exc1_grid[ia]
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
        if mf_grad.grid_response:
            log.debug('grids response contributions')
            _write(log, mol, vhf.exc1_grid[atmlst], atmlst)
            log.debug1('sum(de) %s', vhf.exc1_grid.sum(axis=0))
    return de

def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if mf.nlc != '':
        if ks_grad.nlcgrids is not None:
            nlcgrids = ks_grad.nlcgrids
        else:
            nlcgrids = mf.nlcgrids
        if nlcgrids.coords is None:
            nlcgrids.build(with_non0tab=True)
    if grids.coords is None:
        grids.build(with_non0tab=True)

    ni = mf._numint
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = get_vxc_full_response(ni, mol, nlcgrids,
                                               mf.xc+'__'+mf.nlc, dm,
                                               max_memory=max_memory,
                                               verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.nlc:
            assert 'VV10' in mf.nlc.upper()
            enlc, vnlc = get_vxc(ni, mol, nlcgrids, mf.xc+'__'+mf.nlc, dm,
                                 max_memory=max_memory,
                                 verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)
    return lib.tag_array(vxc, exc1_grid=exc)

def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[0], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc[0]
                aow = numint._scale_ao(ao[0], wv)
                rks_grad._d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:4], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)

    elif xctype == 'NLC':
        nlc_pars = ni.nlc_coeff(xc_code)
        ao_deriv = 2
        vvrho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            vvrho.append([make_rho(idm, ao[:4], mask, 'GGA')
                          for idm in range(nset)])

        vv_vxc = []
        for idm in range(nset):
            rho = numpy.hstack([r[idm] for r in vvrho])
            vxc = numint._vv10nlc(rho, grids.coords, rho, grids.weights,
                                  grids.coords, nlc_pars)[1]
            vv_vxc.append(xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0))

        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            p0, p1 = p1, p1 + weight.size
            for idm in range(nset):
                wv = vv_vxc[idm][:,p0:p1] * weight
                wv[0] *= .5  # *.5 because vmat + vmat.T at the end
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao[:10], mask, xctype)
                vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                rks_grad._gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
                rks_grad._tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat

class Gradients(grad.rks.Gradients):

    grad_elec = grad_elec
    get_veff = get_veff




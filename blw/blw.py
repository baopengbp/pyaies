#
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Block Localized Wavefunction(BLW) ref: Bao, P.; Hettich, C. P.; Shi, Q.; Gao, J. J. Chem. Theory Comput 2021, 17, 240.

import sys
import tempfile

from functools import reduce
import numpy
import scipy.linalg
import h5py
from pyscf import gto, scf, dft
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import diis
from pyscf.scf import _vhf
from pyscf.scf import chkfile
from pyscf.scf import dispersion
from pyscf.data import nist
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)
MUTE_CHKFILE = getattr(__config__, 'scf_hf_SCF_mute_chkfile', False)

class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None, Corth=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = 0
        self.space = 8
        self.Corth = Corth

    def update(self, f, errvec, *args, **kwargs):
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

_err = []
_vec = []
class BLWDIIS(object):
    def __init__(self):
      self.space = 8
    def update(self, f, errvec, *args, **kwargs):
      nblk = len(f)
      fnew = []
      for k in range(nblk):
        _err.append([])
        _vec.append([])
        _err[k].insert(0,errvec[k])
        _vec[k].insert(0,f[k])
        if len(_err[k]) > self.space:
            _err[k].pop()
            _vec[k].pop() 
        nd = len(_err[k])
     
      h = numpy.zeros((nd+1,nd+1))
      h[:,0] = 1.0
      h[0,:] = 1.0
      h[0,0] = 0.0   
      for i in range(nd):
        for j in range(i+1):
            for k in range(nblk): 
                h[i+1,j+1] += _err[k][i]@_err[k][j]
            h[j+1,i+1] = h[i+1,j+1]

      g = numpy.zeros(nd+1)
      g[0] = 1.0
      col = numpy.linalg.solve(h, g)
      for k in range(nblk):
        fnew.append(numpy.zeros(f[k].shape))
        for i, ci in enumerate(col[1:]):
            fnew[k] += ci * _vec[k][i]

      return fnew

def make_t_(tb, s1e):
    stb = tb.T@s1e@tb
    t_ = tb@(numpy.linalg.inv(stb)) 
    return t_

def make_nodm(tb, s1e):
    db = make_t_(tb, s1e)@tb.T 
    return db

def rblw_init(mf, h1e=None, s1e=None):
    if h1e is None: h1e = mf.get_hcore()
    nao = s1e.shape[0]
    occ_blk = [0]
    occbn = 0
    nao_blk = [0]
    naobn = 0
    for i in range(len(mf.block)):
        occbn += mf.block[i][0]//2
        occ_blk.append(occbn)
        naobn += mf.block[i][1]
        nao_blk.append(naobn)
    mo_energy, mo_coeff = mf.eig(h1e, s1e)
    mo_coeff[:, :occ_blk[-1]] = numpy.zeros((nao,occ_blk[-1]))

    for i in range(len(mf.block)):
        sa = s1e[nao_blk[i]:nao_blk[i+1],nao_blk[i]:nao_blk[i+1]]
        ha = h1e[nao_blk[i]:nao_blk[i+1],nao_blk[i]:nao_blk[i+1]]
        mo_energya, mo_coeffa = mf.eig(ha, sa)
        mo_coeff[nao_blk[i]:nao_blk[i+1],occ_blk[i]:occ_blk[i+1]] = mo_coeffa[:,:mf.block[i][0]//2]

    return mo_coeff

def rblw(mf, f, s1e, dm, mo_coeff):

    nao = f.shape[-1]
    iu = numpy.eye(nao)   
    mo_energy = numpy.ones(nao)
    occ_blk = [0]
    occbn = 0
    nao_blk = [0]
    naobn = 0
    for i in range(len(mf.block)):
        occbn += mf.block[i][0]//2
        occ_blk.append(occbn)
        naobn += mf.block[i][1]
        nao_blk.append(naobn)
    mo_coeff_occ = numpy.zeros_like(mo_coeff[:, :occ_blk[-1]])

    fa = []
    sa = []
    fs = []
    grad = []
    for i in range(len(mf.block)):
        s1e_a = s1e[:,nao_blk[i]:nao_blk[i+1]]
        ta = mo_coeff[:,occ_blk[i]:occ_blk[i+1]]
        tba = mo_coeff[:,:occ_blk[i]]
        tbb = mo_coeff[:,occ_blk[i+1]:occ_blk[-1]]
        if tba.shape[1]==0:
            tb = tbb
        elif tbb.shape[1]==0:
            tb = tba
        else:
            tb = numpy.hstack((tba,tbb))
        db = make_nodm(tb, s1e)  
    
        ia = iu[:,nao_blk[i]:nao_blk[i+1]]
        pa = ia - db@s1e_a
        sa.append(s1e_a.T@pa)
        fa.append(pa.T@f@pa)
        # diis err
        taa = ta[nao_blk[i]:nao_blk[i+1]]
        _grad = (ia[nao_blk[i]:nao_blk[i+1]]-sa[i]@taa@taa.T)@fa[i]@taa
        grad.append(_grad.ravel())
        fs.append(numpy.vstack((fa[i], sa[i])))
        mo_coeff_occ[:,occ_blk[i]:occ_blk[i+1]] = ta

    t_ = make_t_(mo_coeff_occ, s1e)
    dm = t_@mo_coeff_occ.T
    xerr = (iu - s1e@dm)@f@t_
    grad1 = []
    for i in range(len(mf.block)):
        grad1.append(xerr[nao_blk[i]:nao_blk[i+1],occ_blk[i]:occ_blk[i+1]].ravel())
    fs = BLWDIIS().update(fs, grad1)

    for i in range(len(mf.block)):
        fa[i], sa[i] = numpy.vsplit(fs[i], 2)
        mo_energya, mo_coeffa = mf.eig(fa[i], sa[i])
        mo_coeff_occ[nao_blk[i]:nao_blk[i+1],occ_blk[i]:occ_blk[i+1]] = mo_coeffa[:,:mf.block[i][0]//2]
        mo_energy[occ_blk[i]:occ_blk[i+1]] = mo_energya[:mf.block[i][0]//2]

    mo_coeff[:, :occ_blk[-1]] = mo_coeff_occ
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = make_nodm(mo_coeff_occ, s1e)
    dm *= 2.0

    grad1 = numpy.hstack(grad1)

    return dm, mo_coeff, mo_occ, grad1, mo_energy

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    '''kernel: the SCF driver.

    Args:
        mf : an instance of SCF class
            mf object holds all parameters to control SCF.  One can modify its
            member functions to change the behavior of SCF.  The member
            functions which are called in kernel are

            | mf.get_init_guess
            | mf.get_hcore
            | mf.get_ovlp
            | mf.get_veff
            | mf.get_fock
            | mf.get_grad
            | mf.eig
            | mf.get_occ
            | mf.make_rdm1
            | mf.energy_tot
            | mf.dump_chk

    Kwargs:
        conv_tol : float
            converge threshold.
        conv_tol_grad : float
            gradients converge threshold.
        dump_chk : bool
            Whether to save SCF intermediate results in the checkpoint file
        dm0 : ndarray
            Initial guess density matrix.  If not given (the default), the kernel
            takes the density matrix generated by ``mf.get_init_guess``.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
        sap_basis : str
            SAP basis name

    Returns:
        A list :   scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

        scf_conv : bool
            True means SCF converged
        e_tot : float
            Hartree-Fock energy of last iteration
        mo_energy : 1D float array
            Orbital energies.  Depending the eig function provided by mf
            object, the orbital energies may NOT be sorted.
        mo_coeff : 2D array
            Orbital coefficients.
        mo_occ : 1D array
            Orbital occupancies.  The occupancies may NOT be sorted from large
            to small.

    Examples:

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=numpy.eye(mol.nao_nr()))
    >>> print('conv = %s, E(HF) = %.12f' % (conv, e))
    conv = True, E(HF) = -1.081170784378
    '''
    if 'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e, **kwargs)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    fock_last = None
    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        #fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis, fock_last=fock_last)
        #mo_energy, mo_coeff = mf.eig(fock, s1e)
        #mo_occ = mf.get_occ(mo_energy, mo_coeff)
        #dm = mf.make_rdm1(mo_coeff, mo_occ) 
        # for blw           
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        if cycle == 0:
            #mo_coeff = rblw_init(mf, h1e, s1e) # orig blw
            mo_coeff = rblw_init(mf, fock, s1e)   
        dm, mo_coeff, mo_occ, grad1, mo_energy = rblw(mf, fock, s1e, dm, mo_coeff)

        #vhf = mf.get_veff(mol, dm, dm_last, vhf)
        vhf = mf.get_veff(mol, dm, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        #norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        # for blw
        norm_gorb = numpy.linalg.norm(grad1)

        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    mf.cycles = cycle + 1
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        ##fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        #mo_energy, mo_coeff = mf.eig(fock, s1e)
        #mo_occ = mf.get_occ(mo_energy, mo_coeff)
        #dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        # for blw
        dm, mo_coeff, mo_occ, grad1, mo_energy = rblw(mf, fock, s1e, dm, mo_coeff)

        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        #norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        # for blw
        norm_gorb = numpy.linalg.norm(grad1)

        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        else:
            scf_conv = False
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def scf(self, dm0=None, **kwargs):
        '''SCF main driver

        Kwargs:
            dm0 : ndarray
                If given, it will be used as the initial guess density matrix

        Examples:

        >>> import numpy
        >>> from pyscf import gto, scf
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1')
        >>> mf = scf.hf.SCF(mol)
        >>> dm_guess = numpy.eye(mol.nao_nr())
        >>> mf.kernel(dm_guess)
        converged SCF energy = -98.5521904482821
        -98.552190448282104
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = self.make_rdm1()

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
        else:
            # Avoid to update SCF orbitals in the non-SCF initialization
            # (issue #495).  But run regular SCF for initial guess if SCF was
            # not initialized.
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot

def blw(mf):
  print('\n================== BLW =========================\n')
  mol = mf.mol
  scf(mf)
  #tools.dump_mat.dump_mo(mol, mf.mo_coeff)
  return 


if __name__ == '__main__':

  mol = gto.M(atom='''
CL      0.000000    0.000000    2.027915
C       0.000000    0.000000   -1.256442
H       0.000000    1.033601   -0.912239
H       0.895125   -0.516801   -0.912239
H      -0.895125   -0.516801   -0.912239
F       0.000000    0.000000   -2.688799
''',
            charge=-1,
#            spin=2,
#            unit = 'B',
            cart = True,
            verbose = 4,
            basis='6-31+g*')
  print('basis=',mol.basis,'nao',mol.nao)
  lib.logger.TIMER_LEVEL = 0

  mf = dft.RKS(mol)
  mf.xc = 'b3lyp5'
  mf.kernel()

  mf = dft.RKS(mol)
  mf.xc = 'b3lyp5'
  mf.block = [[26,48],[10,19]]
  blw(mf)





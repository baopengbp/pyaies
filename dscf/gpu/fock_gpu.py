from functools import reduce
import numpy as np
import cupy
from pyscf.scf import uhf
from pyscf import lib as pyscf_lib
from pyscf import __config__

from gpu4pyscf.scf.hf import eigh, damping, level_shift
from gpu4pyscf.scf import hf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import diis

def uhf_lvs(mf):

  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (level_shift(s1e, dm[0], f[0], shifta).get(),
             level_shift(s1e, dm[1], f[1], shiftb).get())
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

def uhf_vlv(mf):

  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        if mf.mol.spin == 0:
            f = (vlv(mf, s1e, f[0], shifta).get(),
                 level_shift(s1e, dm[1], f[1], shiftb).get()) 
        elif mf.mol.spin == 2:  
            f = (vlvta(mf, s1e, f[0], shifta).get(),
                 vlvtb(mf, s1e, f[1], shiftb).get())          
        else:
            print('To do if not 0 or 2 !111111111')
            exit()
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

from gpu4pyscf import grad
def make_rdm1e(mf_grad, mo_energy, mo_coeff, mo_occ):
    # like rohf
    '''Energy weighted density matrix'''
    mf = mf_grad.base
    fa, fb = mf.focka, mf.fockb
    mocc_a = mo_coeff[0][:,mo_occ[0]>0 ]
    mocc_b = mo_coeff[1][:,mo_occ[1]>0]
    rdm1e_a = reduce(cupy.dot, (mocc_a, mocc_a.conj().T, fa, mocc_a, mocc_a.conj().T))
    rdm1e_b = reduce(cupy.dot, (mocc_b, mocc_b.conj().T, fb, mocc_b, mocc_b.conj().T))
    return cupy.array((rdm1e_a, rdm1e_b))
grad.uhf.Gradients.make_rdm1e = make_rdm1e

def lowdin(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v = cupy.linalg.eigh(s)
    idx = e > 1e-15
    return cupy.dot(v[:,idx]/cupy.sqrt(e[idx]), v[:,idx].conj().T)

def uno(dmt,s1e):
        #s1e**(-1/2)
        issv = lowdin(s1e)
        ssv = s1e@issv
        dmt = ssv.T@dmt@ssv
        e_a, c_a = cupy.linalg.eigh(dmt)
        a= cupy.argsort(e_a)
        a = a[::-1]
        xx = e_a[a]
        c_a = issv@c_a
        c_a = c_a[:,a] 
        return c_a, xx

def cuhf0(mf,fock, dm):
        mol = mf.mol
        s1e = mf.get_ovlp(mol)
        dmt = dm[0]+dm[1]
        # natural orbital
        c_a, e_a = uno(dmt, s1e)
        lmd = c_a.T@(fock[0] - fock[1])@c_a       
        clmd = cupy.zeros_like(lmd) 
        if mol.spin != 0:
            nb = mol.nelec[1]
            na = mol.nelec[0]
            clmd[:nb, na:] = -0.5*lmd[:nb, na:]
        # for 1e excited mix state not ground state
        else:
            nc = mol.nelectron // 2 - 1
            clmd[:nc, nc+2:] = -0.5*lmd[:nc, nc+2:]
            clmd[nc, nc+1] -= 0.5*lmd[nc, nc+1]

        clmd += clmd.T
        sv = s1e@c_a
        nlmd = sv@clmd@sv.T
        fock[0] += nlmd
        fock[1] -= nlmd
        return fock

def cuhf0_fcs(mf,fock, dm):
        mol = mf.mol
        s1e = mf.get_ovlp(mol)
        dmt = dm[0]+dm[1]
        # natural orbital
        c_a, e_a = uno(dmt, s1e)
        print('tt',e_a[:10])
        lmd = c_a.T@(fock[0] - fock[1])@c_a       
        clmd = cupy.zeros_like(lmd) 
        if mol.spin != 0:
            nb = mol.nelec[1]
            na = mol.nelec[0]
            clmd[:nb, na:] = -0.5*lmd[:nb, na:]
        # for 1e excited mix state not ground state
        else:
            nc = mol.nelectron // 2 - 1
            clmd[:nc, nc+2:] = -0.5*lmd[:nc, nc+2:]
            clmd[nc:nc+2, nc:nc+2] = -0.25*lmd[nc:nc+2, nc:nc+2]
            clmd[:nc, :nc] = -0.25*lmd[:nc, :nc]
            clmd[nc+2:, nc+2:] = -0.25*lmd[nc+2:, nc+2:]

        clmd += clmd.T
        sv = s1e@c_a
        nlmd = sv@clmd@sv.T
        fock[0] += nlmd
        fock[1] -= nlmd
        return fock

def cuhf_fcs(mf):
  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    #print('tttttttttttttt')
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)

    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()
    # cuhf
    f = cuhf0_fcs(mf, f, dm) 

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (level_shift(s1e, dm[0], f[0], shifta).get(),
             level_shift(s1e, dm[1], f[1], shiftb).get())
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

def cuhf(mf):
  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()
    # cuhf
    f = cuhf0(mf, f, dm) 

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (level_shift(s1e, dm[0], f[0], shifta).get(),
             level_shift(s1e, dm[1], f[1], shiftb).get())
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

def cuhf_vlv(mf):

  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()
    # cuhf
    f = cuhf0(mf, f, dm) 

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        if mf.mol.spin == 0:
            f = (vlv(mf, s1e, f[0], shifta).get(),
                 level_shift(s1e, dm[1], f[1], shiftb).get())
        elif mf.mol.spin == 2: 
            f = (vlvta(mf, s1e, f[0], shifta).get(),
                 vlvtb(mf, s1e, f[1], shiftb).get())          
        else:
            print('To do if not 0 or 2 !111111111')
            exit()
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

def cuhf_vlv_fcs(mf):

  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = cupy.asarray(mf.get_hcore())
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    if not isinstance(s1e, cupy.ndarray): s1e = cupy.asarray(s1e)
    if not isinstance(dm, cupy.ndarray): dm = cupy.asarray(dm)
    if not isinstance(h1e, cupy.ndarray): h1e = cupy.asarray(h1e)
    if not isinstance(vhf, cupy.ndarray): vhf = cupy.asarray(vhf)
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    # like rohf
    mf.focka = f[0].copy()
    mf.fockb = f[1].copy()
    # cuhf
    f = cuhf0_fcs(mf, f, dm) 

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (damping(s1e, dm[0], f[0], dampa),
             damping(s1e, dm[1], f[1], dampb))
    # for lvs
    if abs(shifta)+abs(shiftb) > 1e-4:
        if mf.mol.spin == 0:
            f = (vlv(mf, s1e, f[0], shifta).get(),
                 level_shift(s1e, dm[1], f[1], shiftb).get())
        elif mf.mol.spin == 2: 
            f = (vlvta(mf, s1e, f[0], shifta).get(),
                 vlvtb(mf, s1e, f[1], shiftb).get())          
        else:
            print('To do if not 0 or 2 !111111111')
            exit()
        f = cupy.asarray(f)

    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    return f

  mf.get_fock = get_fock 
  return mf

def vlv(mf, s, f, shift):
        th = mf.mo_last[0][:,mf.hh:]
        th1 = mf.mo_last[0][:,mf.hh+1:]
        tp = mf.mo_last[0][:,mf.pp:]
        tp1 = mf.mo_last[0][:,mf.pp+1:]
        homo = mf.mol.nelec[0] - 1
        lumo = homo + 1
        tl = mf.mo_last[0][:,lumo:]

        f += shift * s@(th@(th.T) + th1@(th1.T) + tl@(tl.T) + tp@(tp.T) + tp1@(tp1.T))@s
        return f

def vlvta(mf, s, f, shift):
        th = mf.mo_last[0][:,mf.hh:]
        th1 = mf.mo_last[0][:,mf.hh+1:]
        tp = mf.mo_last[0][:,mf.pp:]
        tp1 = mf.mo_last[0][:,mf.pp+1:]
        homo = mf.mol.nelec[0] - 1
        lumo = homo + 1
        tl = mf.mo_last[0][:,lumo:]

        dm = tp1@(tp1.T)
        if mf.pp != lumo:
          dm += tp@(tp.T) + tl@(tl.T) 
        else:
          dm += tp@(tp.T) 
        #f += shifta * s@(th@(th.T) + th1@(th1.T) + tp@(tp.T) + tp1@(tp1.T))@s
        f += shift * s@dm@s
        return f

def vlvtb(mf, s, f, shift):
        th = mf.mo_last[1][:,mf.hh:]
        th1 = mf.mo_last[1][:,mf.hh+1:]
        tp = mf.mo_last[1][:,mf.pp:]
        tp1 = mf.mo_last[1][:,mf.pp+1:]
        homo = mf.mol.nelec[1] - 1
        lumo = homo + 1
        tl = mf.mo_last[1][:,lumo:]

        if mf.hh == homo:
          dm = th@(th.T)
        else:
          dm = th1@(th1.T) + tl@(tl.T) 
          if mf.hh != 0:
            dm += th@(th.T)
        if mf.h1l is True:
          dm = th1@(th1.T)
        #f += shifta * s@(th@(th.T) + th1@(th1.T) + tp@(tp.T) + tp1@(tp1.T))@s
        f += shift * s@dm@s
        return f

from functools import reduce
def get_roothaan_fock(focka_fockb, dma_dmb, s):
    '''Roothaan's effective fock.
    Ref. http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html

    ======== ======== ====== =========
    space     closed   open   virtual
    ======== ======== ====== =========
    closed      Fc      Fb     Fc
    open        Fb      Fc     Fa
    virtual     Fc      Fa     Fc
    ======== ======== ====== =========

    where Fc = (Fa + Fb) / 2

    Returns:
        Roothaan effective Fock matrix
    '''
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc = cupy.dot(dmb, s)
    po = cupy.dot(dma-dmb, s)
    pv = cupy.eye(nao) - cupy.dot(dma, s)
    fock  = reduce(cupy.dot, (pc.conj().T, fc, pc)) * .5
    fock += reduce(cupy.dot, (po.conj().T, fc, po)) * .5
    fock += reduce(cupy.dot, (pv.conj().T, fc, pv)) * .5
    fock += reduce(cupy.dot, (po.conj().T, fockb, pc))
    fock += reduce(cupy.dot, (po.conj().T, focka, pv))
    fock += reduce(cupy.dot, (pv.conj().T, fc, pc))
    fock = fock + fock.conj().T
    fock = tag_array(fock, focka=focka, fockb=fockb)
    return fock

def get_roothaan_fock_4(focka_fockb, dma_dmb, s):
    '''Roothaan's effective fock. for oa and ob

       c  a  b  v
    c Fc Fb Fa Fc
    a Fb Fa Fc Fa
    b Fa Fc Fb Fb
    v Fc Fa Fb Fc

    where Fc = (Fa + Fb) / 2

    Returns:
        Roothaan effective Fock matrix
    '''
    print('4*4 ab fock')
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc  = dma.dot(s).dot(dmb).dot(s)        
    poa = dma.dot(s) - pc                 
    pob = dmb.dot(s) - pc               
    pv  = cupy.eye(nao) - (pc + poa + pob) 
    fock  = reduce(cupy.dot, (pc.conj().T,  fc, pc)) * .5
    fock += reduce(cupy.dot, (poa.conj().T, focka, poa)) * .5
    fock += reduce(cupy.dot, (pob.conj().T, fockb, pob)) * .5
    fock += reduce(cupy.dot, (pv.conj().T,  fc, pv)) * .5
    fock += reduce(cupy.dot, (poa.conj().T, fockb, pc))   
    fock += reduce(cupy.dot, (pob.conj().T, focka, pc))   
    fock += reduce(cupy.dot, (poa.conj().T, focka, pv))   
    fock += reduce(cupy.dot, (pob.conj().T, fockb, pv))   
    fock += reduce(cupy.dot, (pv.conj().T,  fc, pc))   
    fock += reduce(cupy.dot, (poa.conj().T, fc, pob))  
    fock = fock + fock.conj().T
    fock = tag_array(fock, focka=focka, fockb=fockb)
    return fock

def get_roothaan_fock_4fc(focka_fockb, dma_dmb, s):
    '''Roothaan's effective fock. for oa and ob

       c  a  b  v
    c Fc Fb Fa Fc
    a Fb Fc Fc Fa
    b Fa Fc Fc Fb
    v Fc Fa Fb Fc

    where Fc = (Fa + Fb) / 2

    Returns:
        Roothaan effective Fock matrix
    '''
    #print('4*4 ab fock')
    nao = s.shape[0]
    focka, fockb = focka_fockb
    dma, dmb = dma_dmb
    fc = (focka + fockb) * .5
# Projector for core, open-shell, and virtual
    pc  = dma.dot(s).dot(dmb).dot(s)        
    poa = dma.dot(s) - pc                 
    pob = dmb.dot(s) - pc               
    pv  = cupy.eye(nao) - (pc + poa + pob) 
    fock  = reduce(cupy.dot, (pc.conj().T,  fc, pc)) * .5
    fock += reduce(cupy.dot, (poa.conj().T, fc, poa)) * .5
    fock += reduce(cupy.dot, (pob.conj().T, fc, pob)) * .5
    fock += reduce(cupy.dot, (pv.conj().T,  fc, pv)) * .5
    fock += reduce(cupy.dot, (poa.conj().T, fockb, pc))   
    fock += reduce(cupy.dot, (pob.conj().T, focka, pc))   
    fock += reduce(cupy.dot, (poa.conj().T, focka, pv))   
    fock += reduce(cupy.dot, (pob.conj().T, fockb, pv))   
    fock += reduce(cupy.dot, (pv.conj().T,  fc, pc))   
    fock += reduce(cupy.dot, (poa.conj().T, fc, pob))  
    fock = fock + fock.conj().T
    fock = tag_array(fock, focka=focka, fockb=fockb)
    return fock

from gpu4pyscf.scf import rohf
def get_fock_ro(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        '''Build fock matrix based on Roothaan's effective fock.
        See also :func:`get_roothaan_fock`
        '''
        if h1e is None: h1e = self.get_hcore()
        if s1e is None: s1e = self.get_ovlp()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = cupy.repeat(dm[None]*.5, 2, axis=0)
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        f = get_roothaan_fock((focka,fockb), dm, s1e)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp

        dm_tot = dm[0] + dm[1]
        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            raise NotImplementedError('ROHF Fock-damping')

        if abs(level_shift_factor) > 1e-4:
            f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm_tot, f, self, h1e, vhf, f_prev=fock_last)

        f = tag_array(f, focka=focka, fockb=fockb)
        return f
rohf.ROHF.get_fock = get_fock_ro 

def rohf_vlv(mf):
  def get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        '''Build fock matrix based on Roothaan's effective fock.
        See also :func:`get_roothaan_fock`
        '''
        if h1e is None: h1e = mf.get_hcore()
        if s1e is None: s1e = mf.get_ovlp()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)
        if dm is None: dm = mf.make_rdm1()
        if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
            dm = cupy.repeat(dm[None]*.5, 2, axis=0)
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        print('rrr',dm.shape)
        f = rohf.get_roothaan_fock((focka,fockb), dm, s1e)
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f

        if diis_start_cycle is None:
            diis_start_cycle = mf.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = mf.level_shift
        if damp_factor is None:
            damp_factor = mf.damp

        dm_tot = dm[0] + dm[1]
        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            raise NotImplementedError('ROHF Fock-damping')

        if abs(level_shift_factor) > 1e-4:
            f = vlv_ro(mf, s1e, f, level_shift_factor)

        if diis and cycle >= diis_start_cycle:
            f = diis.update(s1e, dm_tot, f, mf, h1e, vhf, f_prev=fock_last)

        f = tag_array(f, focka=focka, fockb=fockb)
        return f

  mf.get_fock = get_fock 
  return mf

def vlv_ro(mf, s, f, shift):
        th = mf.mo_last[:,mf.hh:]
        th1 = mf.mo_last[:,mf.hh+1:]
        tp = mf.mo_last[:,mf.pp:]
        tp1 = mf.mo_last[:,mf.pp+1:]
        homo = mf.mol.nelec[0] - 1
        lumo = homo + 1
        tl = mf.mo_last[:,lumo:]

        dm = tp1@(tp1.T)
        if mf.pp == lumo:
          dm += th1@(th1.T)
          if mf.hh != 0:
            dm += th@(th.T)
        else:
          dm += tp@(tp.T)
          if mf.hh == homo:
            if mf.hh != 0:
              dm += th@(th.T)
          else:
            dm += tl@(tl.T) + th1@(th1.T)
            if mf.hh != 0:
              dm += th@(th.T)
        f += shift * s@dm@s
        return f









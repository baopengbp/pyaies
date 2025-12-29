#!/usr/bin/env python

import numpy, scipy
from pyscf import gto, scf, dft
from pyscf import lib
lib.logger.TIMER_LEVEL = 0
from pyscf.scf import uhf
from pyscf.lo import orth

def uno(dmt,s1e):
        #s1e**(-1/2)
        issv = orth.lowdin(s1e)
        ssv = s1e@issv
        dmt = ssv.T@dmt@ssv
        e_a, c_a = scipy.linalg.eigh(dmt)
        a= numpy.argsort(e_a)
        a = a[::-1]
        xx = e_a[a]
        c_a = issv@c_a
        c_a = c_a[:,a] 
        return c_a, xx

def cuhf(mf,fock, dm):
        mol = mf.mol
        s1e = mf.get_ovlp(mol)
        dmt = dm[0]+dm[1]
   
        c_a, e_a = uno(dmt, s1e)
        #print('tt',e_a)
        #print('fock',fock)
        lmd = c_a.T@(fock[0] - fock[1])@c_a
        
        clmd = numpy.zeros_like(lmd) #((N,N))
        clmd[:mol.nelec[1], mol.nelec[0]:] = -0.5*lmd[:mol.nelec[1], mol.nelec[0]:]
        clmd += clmd.T
        #print('ee',clmd)
        sv = s1e@c_a
        lmd = sv@clmd@sv.T
        fock[0] += lmd
        fock[1] -= lmd
        #print('hhh',fock.shape)
        #exit()
        return fock

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    f = numpy.asarray(h1e) + vhf
    if f.ndim == 2:
        f = (f, f)
    #########
    f = cuhf(mf, f, dm) 

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

    if isinstance(level_shift_factor, (tuple, list, numpy.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, numpy.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (hf.damping(s1e, dm[0], f[0], dampa),
             hf.damping(s1e, dm[1], f[1], dampb))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (hf.level_shift(s1e, dm[0], f[0], shifta),
             hf.level_shift(s1e, dm[1], f[1], shiftb))
    return numpy.array(f)

class CUHF(uhf.UHF):

  def __init__(self, mol):
      uhf.UHF.__init__(self, mol) 
      self.mo0 = None
  get_fock = get_fock

class CUKS(dft.uks.UKS):

  def __init__(self, mol):
      dft.uks.UKS.__init__(self, mol) 
      self.mo0 = None
  get_fock = get_fock

if __name__ == '__main__':

  mol = gto.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
''',
            charge=-1,
            spin=1,
#            unit = 'B',
            #cart = True,
            verbose = 4,
            basis='6-31g')
  print('basis=',mol.basis,'nao',mol.nao)

#uks
  a = dft.UKS(mol)
  a.xc = 'pbe0'
  ae = a.scf()

  b = CUKS(mol)
  #b.init_guess = '1e'
  b.xc = 'pbe0'
  b.max_cycle = 15
  be = b.scf()

  a = dft.ROKS(mol)
  a.xc = 'pbe0'
  ae = a.scf()



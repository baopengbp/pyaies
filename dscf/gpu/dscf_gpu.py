#
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Delta-SCF

import numpy
import cupy
from functools import reduce
from pyscf import gto, scf, dft, lib
from pyscf.lib import logger
import cupy
import cuhf
import fock_gpu
from gpu4pyscf import scf as scf_gpu
from gpu4pyscf import dft as dft_gpu
from pyscf.scf import hf as hf_cpu

def det_ovlp(mo1, mo2, occ1, occ2, ovlp):
    ovlp = cupy.asarray(ovlp)
    c1_a = mo1[0][:, occ1[0]>0]
    c1_b = mo1[1][:, occ1[1]>0]
    c2_a = mo2[0][:, occ2[0]>0]
    c2_b = mo2[1][:, occ2[1]>0]
    o_a = cupy.asarray(reduce(cupy.dot, (c1_a.conj().T, ovlp, c2_a)))
    o_b = cupy.asarray(reduce(cupy.dot, (c1_b.conj().T, ovlp, c2_b)))

    det_ovlp = cupy.linalg.det(o_a) * cupy.linalg.det(o_b)   

    u_a, s_a, vt_a = cupy.linalg.svd(o_a)
    u_b, s_b, vt_b = cupy.linalg.svd(o_b)
    s_a = cupy.where(abs(s_a) > 1.0e-11, s_a, 1.0e-11)
    s_b = cupy.where(abs(s_b) > 1.0e-11, s_b, 1.0e-11) 
    OV = cupy.linalg.det(u_a)*cupy.linalg.det(u_b) \
       * cupy.prod(s_a)*cupy.prod(s_b) \
       * cupy.linalg.det(vt_a)*cupy.linalg.det(vt_b) 
    x_a = reduce(cupy.dot, (u_a*cupy.reciprocal(s_a), vt_a))
    x_b = reduce(cupy.dot, (u_b*cupy.reciprocal(s_b), vt_b))

    return OV, cupy.array((x_a, x_b))

def make_asym_dm(mo1, mo2, occ1, occ2, x):
    mo1_a = mo1[0][:, occ1[0]>0]
    mo1_b = mo1[1][:, occ1[1]>0]
    mo2_a = mo2[0][:, occ2[0]>0]
    mo2_b = mo2[1][:, occ2[1]>0]
    dm_a = reduce(cupy.dot, (mo1_a, x[0], mo2_a.conj().T))
    dm_b = reduce(cupy.dot, (mo1_b, x[1], mo2_b.conj().T))
    return cupy.array((dm_a, dm_b))

def trans_dip(mf, mol, mo0, mo1, occ0, occ1):
  mfhf = scf_gpu.UHF(mol)
  s_ao = cupy.asarray(mol.intor_symmetric('int1e_ovlp'))
# Calculate overlap between two determiant <I|F>
  s, x = det_ovlp(mo0, mo1, occ0, occ1, s_ao)
# Construct density matrix 
  dm_01 = make_asym_dm(mo0, mo1, occ0, occ1, x)

  with mol.with_common_orig(numpy.zeros(3)):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
  if mf.dips == 'Orig':
    #print('&&&&&&& original dipole &&&&&&&')
    dip = s * cupy.einsum('xji,ji->x', ao_dip, dm_01[0]+dm_01[1])
  elif mf.dips == 'Schmidt':
    #print('&&&&&&& dipole of Schmidt ortho &&&&&&&')
    d01 = s * (dm_01[0] + dm_01[1])
    d00_a, d00_b = mfhf.make_rdm1(mo0, occ0)
    d00 = d00_a + d00_b
    dip_g  = cupy.einsum('xji,ji->x', ao_dip, d00)
    dip_ge = cupy.einsum('xji,ji->x', ao_dip, d01)
    dip = (dip_ge - s * dip_g) / numpy.sqrt(1 - s * s)
  else:
    #print('&&&&&&& dipole of lowdin ortho &&&&&&&')
    d01 = s * (dm_01[0] + dm_01[1])
    d00_a, d00_b = mfhf.make_rdm1(mo0, occ0)
    d00 = d00_a + d00_b
    d11_a, d11_b = mfhf.make_rdm1(mo1, occ1)
    d11 = d11_a + d11_b
    a = numpy.sqrt((1+s) / (1-s))
    d01_ = ((1-a*a)*(d00+d11) + (1+a)*(1+a)*d01 + (1-a)*(1-a)*d01.T)/4/(1+s)
    dip = cupy.einsum('xji,ji->x', ao_dip, d01_)
  dip2 = dip.dot(dip)

  return dip2, s

def pd_v_ord(mo0, fock):
        fock = mo0.T@fock@mo0
        mo_energy, coeff = cupy.linalg.eigh(fock)
        mo_coeff = mo0@coeff      
        return mo_energy, mo_coeff

def chp_ord_(mf,fock,s):
        # Hole P'+Ph'
        tt = mf.mo_last[0][:, :mf.mol.nelec[0] + 1]
        d_a = tt@(tt.T)
        na = d_a.shape[0]
        ia = cupy.eye(na)
        d_o = cupy.dot(mf.mo0[0][:,:mf.mol.nelec[0]], mf.mo0[0][:,:mf.mol.nelec[0]].T)
        pj = d_a@s
        fa = pj.T@fock[0]@pj
        e_o,c_o = pd_v_ord(mf.mo0[0][:,:mf.mol.nelec[0]], fa) 

        # Particle P'-Pp'     
        tt = mf.mo_last[0][:, :mf.mol.nelec[0] - 1]
        d_a = tt@(tt.T)
        na = d_a.shape[0]
        ia = cupy.eye(na)
        d_v = cupy.dot(mf.mo0[0][:,mf.mol.nelec[0]:], mf.mo0[0][:,mf.mol.nelec[0]:].T)
        pj = (ia - d_a@s)
        fa = pj.T@fock[0]@pj
        e_vir,c_vir = pd_v_ord(mf.mo0[0][:,mf.mol.nelec[0]:], fa) 

        pp = mf.pp - mf.mol.nelec[0]
        d_a = cupy.outer(c_vir[:,pp],c_vir[:,pp]) + cupy.outer(c_o[:,mf.hh],c_o[:,mf.hh])
        ia = cupy.eye(d_a.shape[0])
        pj = ia - d_a@s
        fa = pj.T@fock[0]@pj
        e_a,c_a = pd_v_ord(mf.mo0[0], fa)
        # put particle in HOMO
        e_a[mf.mol.nelec[0]-1] = -10.0 
        c_a[:, mf.mol.nelec[0]-1] = c_vir[:, pp]
        # put Hole in LUMO
        e_a[mf.mol.nelec[0]] = e_o[mf.hh]
        c_a[:, mf.mol.nelec[0]] = c_o[:, mf.hh]


        e_b,c_b = pd_v_ord(mf.mo0[1], fock[1])    
        mo_energy = cupy.array((e_a,e_b)) 
        mo_coeff = cupy.array((c_a, c_b))

        mf.mo_last = mo_coeff
        return mo_energy, mo_coeff

def chp_ord(mf):
  def eig(h, s):
    return chp_ord_(mf,h,s)
  mf.eig = eig 
  return mf

def cp_ord_(mf,fock,s):
        occ_ne = mf.occ_init[0].copy()
        occ_ne[mf.pp] = 0.0        
        tt = mf.mo_last[0][:, occ_ne > 0]
        d_a = tt@(tt.T)
        na = d_a.shape[0]
        ia = cupy.eye(na)
        d_v = cupy.dot(mf.mo0[0][:,mf.mol.nelec[0]:], mf.mo0[0][:,mf.mol.nelec[0]:].T)
        pj = (ia - d_a@s)@d_v@s
        fa = pj.T@fock[0]@pj
        e_vir,c_vir = pd_v_ord(mf.mo0[0][:,mf.mol.nelec[0]:], fa) #

        pp = mf.pp - mf.mol.nelec[0]
        d_a = cupy.outer(c_vir[:,pp],c_vir[:,pp])
        ia = cupy.eye(d_a.shape[0])
        pj = ia - d_a@s
        fa = pj.T@fock[0]@pj
        e_a,c_a = pd_v_ord(mf.mo0[0], fa)
        e_a[mf.mol.nelec[0]:] = e_vir[:na-mf.mol.nelec[0]]
        c_a[:, mf.mol.nelec[0]:] = c_vir[:, :na-mf.mol.nelec[0]]

        e_b,c_b = pd_v_ord(mf.mo0[1], fock[1])    
        mo_energy = cupy.array((e_a,e_b)) 
        mo_coeff = cupy.array((c_a, c_b))

        mf.mo_last = mo_coeff
        return mo_energy, mo_coeff

def cp_ord(mf):
  def eig(h, s):
    return cp_ord_(mf,h,s)
  mf.eig = eig 
  return mf

def get_gorb(mo_coeff, mo_occ, fock, s):
        ia = cupy.eye(s.shape[0])
        mo_a = mo_coeff[0][:,mo_occ[0]>0]
        ga = (ia - s@mo_a@(mo_a.T))@(fock[0])@mo_a
        mo_b = mo_coeff[1][:,mo_occ[1]>0]
        gb = (ia - s@mo_b@(mo_b.T))@(fock[1])@mo_b
        return cupy.hstack((ga.ravel(), gb.ravel()))

def gorb_occ(mf):
  s = cupy.asarray(mf.get_ovlp())
  def get_grad(mo_coeff, mo_occ, fock_ao):
    '''UHF Gradients'''
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb

    ga = mo_coeff[0][:,viridxa].conj().T.dot(fock_ao[0].dot(mo_coeff[0][:,occidxa]))
    gb = mo_coeff[1][:,viridxb].conj().T.dot(fock_ao[1].dot(mo_coeff[1][:,occidxb]))

    gorb = get_gorb(mo_coeff, mo_occ, fock_ao, s)  
    return gorb
  mf.get_grad = get_grad
  return mf


def mom_occ_cp(mf, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration.'''
    from pyscf.scf import uhf, rohf
    log = logger.Logger(mf.stdout, mf.verbose)
    if mf.istype('UHF'):
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]
    elif mf.istype('ROHF'):
        if mf.mol.spin != (cupy.sum(setocc[0]) - cupy.sum(setocc[1])):
            raise ValueError('Wrong occupation setting for restricted open-shell calculation.')
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    else: # GHF, and DHF
        assert setocc.ndim == 1

    if mf.istype('UHF') or mf.istype('ROHF'):
        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_coeff is None: mo_coeff = mf.mo_coeff
            if mf.istype('ROHF'):
                mo_coeff = cupy.array([mo_coeff, mo_coeff])
            mo_occ = cupy.zeros_like(setocc)
            nocc_a = int(cupy.sum(setocc[0]))
            nocc_b = int(cupy.sum(setocc[1]))
            s_a = reduce(cupy.dot, (coef_occ_a.conj().T, mf.get_ovlp(), mo_coeff[0]))
            s_b = reduce(cupy.dot, (coef_occ_b.conj().T, mf.get_ovlp(), mo_coeff[1]))
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx_a = cupy.argsort(cupy.einsum('ij,ij->j', s_a, s_a))[::-1]
            idx_b = cupy.argsort(cupy.einsum('ij,ij->j', s_b, s_b))[::-1]
            mo_occ[0,idx_a[:nocc_a]] = 1.
            mo_occ[1,idx_b[:nocc_b]] = 1.
            log.debug(' New alpha occ pattern: %s', mo_occ[0])
            log.debug(' New beta occ pattern: %s', mo_occ[1])
            if isinstance(mf.mo_energy, cupy.ndarray) and mf.mo_energy.ndim == 1:
                log.debug1(' Current mo_energy(sorted) = %s', mo_energy)
            else:
                log.debug1(' Current alpha mo_energy(sorted) = %s', mo_energy[0])
                log.debug1(' Current beta mo_energy(sorted) = %s', mo_energy[1])

            if (int(cupy.sum(mo_occ[0])) != nocc_a):
                log.error('mom alpha electron occupation numbers do not match: %d, %d',
                          nocc_a, int(cupy.sum(mo_occ[0])))
            if (int(cupy.sum(mo_occ[1])) != nocc_b):
                log.error('mom beta electron occupation numbers do not match: %d, %d',
                          nocc_b, int(cupy.sum(mo_occ[1])))

            #output 1-dimension occupation number for restricted open-shell
            if mf.istype('ROHF'):
                mf.occa = mo_occ[0, :]
                mf.occb = mo_occ[1, :]                
            return mo_occ
    else:
        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_coeff is None: mo_coeff = mf.mo_coeff
            mo_occ = cupy.zeros_like(setocc)
            nocc = int(setocc.sum())
            s = occorb[:,setocc>0].conj().T.dot(mf.get_ovlp()).dot(mo_coeff)
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx = cupy.argsort(cupy.einsum('ij,ij->j', s, s))[::-1]
            mo_occ[idx[:nocc]] = 1.
            return mo_occ

    mf.get_occ = get_occ
    return mf

def run(mf):
  mol = mf.mol
  s_ao = cupy.asarray(mol.intor_symmetric('int1e_ovlp'))
  mo = []
  occa = []
  e_e = []

  if mf.df is True:
    g = dft.RKS(mol).density_fit(mf.auxbasis).to_gpu()
  else:
    g = dft.RKS(mol).to_gpu()

  #g.level_shift = 0.5
  g.max_cycle = mf.max_cycle
  g.diis_space = 20
  g.xc = mf.xc
  e_g = g.scf()
  g._cderi = None # delocate eri
  print('g_mo_energy',g.mo_energy) 
  #tools.dump_mat.dump_mo(mol, g.mo_coeff[0])
  mo0 = cupy.asarray([g.mo_coeff, g.mo_coeff])
  mo_occ = cupy.asarray([g.mo_occ/2, g.mo_occ/2])
  # s
  e_sm = []
  e_t = []
  s_conv = []
  t_conv = []

  if mf.mthd == 'ROKS' or mf.mthd == 'SFRO':
   for i in range(len(mf.s)):
    mol.spin = 2
    mol.build()
    occ_t = mo_occ.copy()
    occ_t[1][mf.s[i][0]]=0      
    occ_t[0][mf.s[i][1]]=1
    ro_occ = occ_t[0] + occ_t[1]
    print('ro_occ=',ro_occ) 
    if mf.df is True:
      t = dft.ROKS(mol).density_fit(mf.auxbasis).to_gpu()
    else:
      t = dft.ROKS(mol).to_gpu()

    if mf.t1_min is True:
      pass
    elif mf.excite == 'pd':
      import ro_ab_gpu
      import pd_gpu
      def get_occ(mo_energy=None, mo_coeff=None):
        return ro_occ
      t.get_occ = get_occ
    # new
      t.occa = cupy.where(ro_occ==2,1,ro_occ)
      t.occb = cupy.where(ro_occ==2,1,0)
    elif mf.excite == 'vlv':
      import ro_ab_gpu
      t = fock_gpu.rohf_vlv(t)
      t.hh = mf.s[i][0]
      t.pp = mf.s[i][1]
      t.occ_init = occ_t
      t.mo_last = mo0[0]
      def get_occ(mo_energy=None, mo_coeff=None):
        t.mo_last = mo_coeff	
        return ro_occ
      t.get_occ = get_occ
      t.h1l = mf.h1l
    # new
      t.occa = cupy.where(ro_occ==2,1,ro_occ)
      t.occb = cupy.where(ro_occ==2,1,0)
    elif mf.excite == 'mom':
      import ro_ab_gpu
      t = mom_occ_cp(t, mo0[0], occ_t)
    elif mf.excite == 'orig':
      def get_occ(mo_energy=None, mo_coeff=None):
        return ro_occ
      t.get_occ = get_occ
    # new
      t.occa = cupy.where(ro_occ==2,1,ro_occ)
      t.occb = cupy.where(ro_occ==2,1,0)

    t.xc = mf.xc
    t.mo0 = mo0[0]
    t.max_cycle = mf.max_cycle
    t.diis_space = 20
    t.level_shift = mf.excite_factor

    if t.level_shift != 0:
      t.conv_check=False

    dm_t_init = t.make_rdm1(mo0[0], ro_occ)
    e_t.append(t.scf(dm_t_init))
    t_conv.append(t.converged)
    t._cderi = None # delocate eri

    # s
    mol.spin = 0
    mol.build()
    occ = mo_occ.copy()
# Assign initial occupation pattern double excited
    occ[0][mf.s[i][0]]=0    
    occ[0][mf.s[i][1]]=1   
    if mf.df is True:
      s = dft.ROKS(mol).density_fit(mf.auxbasis).to_gpu()
    else:
      s = dft.ROKS(mol).to_gpu()

    if mf.excite == 'pd':
      def get_occ(mo_energy=None, mo_coeff=None):
        return occ
      s.get_occ = get_occ
      s.occa = occ[0]
      s.occb = occ[1]
    elif mf.excite == 'vlv':
      import ro_ab_gpu
      s = fock_gpu.rohf_vlv(s)
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      s.occ_init = occ
      s.mo_last = mo0[0]
      def get_occ(mo_energy=None, mo_coeff=None):
        s.mo_last = mo_coeff	
        return occ
      s.get_occ = get_occ
      s.h1l = mf.h1l 
      s.occa = occ[0]
      s.occb = occ[1]
    elif mf.excite == 'mom':
      import ro_ab_gpu
      s = mom_occ_cp(s, mo0[0], occ)
    elif mf.excite == 'orig':
      import ro_ab_gpu
      def get_occ(mo_energy=None, mo_coeff=None):
        return occ
      s.get_occ = get_occ
      s.occa = occ[0]
      s.occb = occ[1]
    s.xc = mf.xc

    s.mo0 = mo0[0]
    if mf.mthd == 'ROKS':
      s.max_cycle = mf.max_cycle
    elif mf.mthd == 'SFRO':
      s.max_cycle = -1
    s.diis_space = 20
    s.level_shift = mf.excite_factor

    if s.level_shift != 0:
      s.conv_check=False

    dm_sm = s.make_rdm1(t.mo_coeff, occ)
    if mf.init_rodm_s_from_g is True:
      dm_sm = s.make_rdm1(mo0[0], occ)
    esm = s.scf(dm_sm)
    e_sm.append(esm)
    s_conv.append(s.converged)
    s._cderi = None # delocate eri

    mo.append([s.mo_coeff, s.mo_coeff])
    occa.append(occ)
    e_e.append(esm)
    # beta
    mo.append([s.mo_coeff, s.mo_coeff])
    occa.append([occ[1],occ[0]])
    e_e.append(esm)

  elif mf.mthd == 'UKS' or mf.mthd == 'CUKS':
   for i in range(len(mf.s)):
    ro_occ = mo_occ.copy()
    ro_occ[1][mf.s[i][0]]=0      
    ro_occ[0][mf.s[i][1]]=1   

    if mf.df is True:
      t = dft.UKS(mol).density_fit(mf.auxbasis).to_gpu()
    else:
      t = dft.UKS(mol).to_gpu()

    mol.spin = 2
    mol.build()

    t.xc = mf.xc
    if mf.t1_min is True:
      pass
    elif mf.excite == 'pd':
      import pd_gpu
      t.aa = 0
      t.hh = mf.s[i][0]
      t.pp = mf.s[i][1]
      t.occ_init = ro_occ
      t.mo_last = mo0.copy()
      def get_occ(mo_energy=None, mo_coeff=None):
        t.mo_last = mo_coeff
        return ro_occ
      t.get_occ = get_occ
    elif mf.excite == 'pd_chp':
      import pd_gpu
      t.hh = mf.s[i][0]
      t.pp = mf.s[i][1]
      mor = mo0.copy()
      mor[0][:,t.hh] = mo0[0][:,t.mol.nelec[0]-1] 
      mor[0][:,t.mol.nelec[0]-1] = mo0[0][:,t.pp]    
      mor[0][:,t.mol.nelec[0]] = mo0[0][:,t.hh]
      t.mo_last = mor
      t.mor = mor[0]
      def get_occ(mo_energy=None, mo_coeff=None):
        return mo_occ
      t.get_occ = get_occ
      t = gorb_occ(t)
      t.c_vir = mo0[0][:,t.mol.nelec[0]:]
      print('\n!!!!!!!!!! pd_chp for Triplet not implement, to do later !!!!!!!\n')
    elif mf.excite == 'pd_vlv':
      import pd_gpu
      t.hh = mf.s[i][0]
      t.pp = mf.s[i][1]
      t.occ_init = ro_occ
      t.mo_last = mo0.copy()
      def get_occ(mo_energy=None, mo_coeff=None):
        t.mo_last = mo_coeff	
        return ro_occ
      t.get_occ = get_occ
      t.h1l = mf.h1l
    elif mf.excite == 'vlv':
      t.hh = mf.s[i][0]
      t.pp = mf.s[i][1]
      t.occ_init = ro_occ
      t.mo_last = mo0
      def get_occ(mo_energy=None, mo_coeff=None):
        t.mo_last = mo_coeff	
        return ro_occ
      t.get_occ = get_occ
      t.h1l = mf.h1l
    elif mf.excite == 'orig':
      def get_occ(mo_energy=None, mo_coeff=None):
          return ro_occ
      t.get_occ = get_occ
    elif mf.excite == 'mom':
      t = scf.addons.mom_occ(t, mo0, ro_occ)
    elif mf.excite == 'ex-lshift':
      ex_pair = [mf.s[i][0], mf.s[i][1]]
      ll = abs(g.mo_energy[ex_pair[0]] - g.mo_energy[ex_pair[1]]) 
      if isinstance(mf.excite_factor, (tuple, list)):
        t.level_shift = (mf.excite_factor[0], ll + mf.excite_factor[1])  
      else:
        t.level_shift = mf.excite_factor 
      t.conv_check=False

    if (mf.excite == 'vlv' or mf.excite == 'pd_vlv') and mf.t1_min is False:
      if mf.mthd == 'UKS':
        t = fock_gpu.uhf_vlv(t)
      elif mf.mthd == 'CUKS':
        t = fock_gpu.cuhf_vlv(t)
    else:
      if mf.mthd == 'UKS':
        t = fock_gpu.uhf_lvs(t)
      elif mf.mthd == 'CUKS':
        t = fock_gpu.cuhf(t)

    t.level_shift = mf.excite_factor
    t.mo0 = mo0
    t.max_cycle = mf.max_cycle
    t.diis_space = 20

    t.mo0_occa = mo0[0][:, g.mo_occ[0]>0]
    t.mo0_occb = mo0[1][:, g.mo_occ[1]>0]
    t.mo0_vira = mo0[0][:, g.mo_occ[0]<1]
    t.mo0_virb = mo0[1][:, g.mo_occ[1]<1]
    #
    dm_t_init = t.make_rdm1(mo0, ro_occ)
    e_t.append(t.scf(dm_t_init))
    t_conv.append(t.converged)
    t._cderi = None # delocate eri

# singlet state
    if mf.df is True:
      s = dft.UKS(mol).density_fit(mf.auxbasis).to_gpu()
    else:
      s = dft.UKS(mol).to_gpu()

    occ = mo_occ.copy()
# Assign initial occupation pattern double excited
    occ[0][mf.s[i][0]]=0      
    occ[0][mf.s[i][1]]=1      

    mol.spin = 0
    mol.build()

    if mf.excite == 'pd':
      s.aa = 0
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      s.occ_init = occ
      s.mo_last = mo0.copy()
      #s.xc = mf.xc
      def get_occ(mo_energy=None, mo_coeff=None):
        s.mo_last = mo_coeff
        return occ
      s.get_occ = get_occ
    if mf.excite == 'cp_ord':
      s.aa = 0
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      s.occ_init = occ
      s.mo_last = mo0.copy()
      s.mo_last_o = mo0.copy()
      s.mo_p = mo0[0][:,s.pp]
      s.mo_last_nh_v = None
      def get_occ(mo_energy=None, mo_coeff=None):
        return occ
      s.get_occ = get_occ
      s = gorb_occ(s)
      s = cp_ord(s)
    if mf.excite == 'pd_chp':
      import pd_gpu
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      mor = mo0.copy()
      mor[0][:,s.hh] = mo0[0][:,s.mol.nelec[0]-1]
      mor[0][:,s.mol.nelec[0]-1] = mo0[0][:,s.pp]    
      mor[0][:,s.mol.nelec[0]] = mo0[0][:,s.hh]
      s.mo_last = mor
      s.mor = mor[0]
      s.c_vir = mo0[0][:,s.mol.nelec[0]:]
      def get_occ(mo_energy=None, mo_coeff=None):
        s.mo_last = mo_coeff
        return mo_occ
      s.get_occ = get_occ
      s = gorb_occ(s)
    if mf.excite == 'chp_ord':
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      mor = mo0.copy()
      mor[0][:,s.hh] = mo0[0][:,s.mol.nelec[0]-1]
      mor[0][:,s.mol.nelec[0]-1] = mo0[0][:,s.pp]    
      mor[0][:,s.mol.nelec[0]] = mo0[0][:,s.hh]
      s.mo_last = mor
      s.c_o = mo0[0]
      s.c_vir = mo0[0][:,s.mol.nelec[0]:]
      s.mo_p = mo0[0][:,s.pp]
      def get_occ(mo_energy=None, mo_coeff=None):
        s.mo_last = mo_coeff
        return mo_occ
      s.get_occ = get_occ
      s = gorb_occ(s)
      s = chp_ord(s) 
    if mf.excite == 'vlv':
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      s.occ_init = occ
      s.mo_last = mo0
      def get_occ(mo_energy=None, mo_coeff=None):
        xx = max(s.hh-3, 0)
        s.mo_last = mo_coeff	
        return occ
      s.get_occ = get_occ
      s.level_shift = mf.excite_factor
      s.h1l = mf.h1l 
    elif mf.excite == 'mom':
      s.aa = 0
      s.hh = mf.s[i][0]
      s.pp = mf.s[i][1]
      s.occ_init = occ
      s.mo_last = mo0.copy()
      s = scf.addons.mom_occ(s, mo0, occ)
    elif mf.excite == 'orig':
      def get_occ(mo_energy=None, mo_coeff=None):
          return occ
      s.get_occ = get_occ
    elif mf.excite == 'ex-lshift':
      s.ex_pair = [mf.s[i][0], mf.s[i][1]]
      ll = abs(g.mo_energy[s.ex_pair[0]] - g.mo_energy[s.ex_pair[1]]) 
      if isinstance(mf.excite_factor, (tuple, list)):
        s.level_shift = (ll + mf.excite_factor[0], mf.excite_factor[1])  
      else:
        s.level_shift = mf.excite_factor #(ll + mf.excite_factor, 0.1)
      print(s.ex_pair[0], 'new s.level_shift',s.level_shift)
      s.conv_check=False
    elif mf.excite == 'levs':
      s = levs
      s.level_shift = mf.excite_factor
      print('s.level_shift =', s.level_shift)
      s.conv_check=False
    if mf.excite == 'vlv' or mf.excite == 'pd_vlv':
      if mf.mthd == 'UKS':
        s = fock_gpu.uhf_vlv(s)
      elif mf.mthd == 'CUKS':
        s = fock_gpu.cuhf_vlv(s)
    else:
      if mf.mthd == 'UKS':
        s = fock_gpu.uhf_lvs(s)
      elif mf.mthd == 'CUKS':
        s = fock_gpu.cuhf(s)

    s.xc = mf.xc
    s.max_cycle = mf.max_cycle
    s.diis_space = 20
    s.mo0 = mo0

    s.mo0_occa = mo0[0][:, mo_occ[0]>0.5]
    s.mo0_occb = mo0[1][:, mo_occ[1]>0.5]
    s.mo0_vira = mo0[0][:, mo_occ[0]<0.5]
    s.mo0_virb = mo0[1][:, mo_occ[1]<0.5]

    dm_sm = s.make_rdm1(mo0, occ)
    esm = s.scf(dm_sm)

    ex_mthd = mf.excite

    e_sm.append(esm)
    s_conv.append((s.converged,ex_mthd))
    s._cderi = None # delocate eri

    mo.append(s.mo_coeff)
    occa.append(s.mo_occ)
    e_e.append(esm)
    # beta
    mo.append([s.mo_coeff[1], s.mo_coeff[0]])
    occa.append([s.mo_occ[1],s.mo_occ[0]])
    e_e.append(esm)
  else:
    print('Error mf.mthd, exit')
    exit() 

  # next d
  e_d = []
  d_conv = []
  for i in range(len(mf.d)):
    #occ = g.mo_occ.copy()
    occ = mo_occ.copy()

# Assign initial occupation pattern double excited
    occ[0][mf.d[i][0]]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
    occ[0][mf.d[i][1]]=1      # 
    occ[1][mf.d[i][0]]=0      # beta
    occ[1][mf.d[i][1]]=1 
    print('occ=',occ) 
# New SCF caculation 
    #d = dft.UKS(mol)
    #d = pdscf.UKS(mol)
    if mf.df is True:
      d = dft.UKS(mol).density_fit(mf.auxbasis).to_gpu()
    else:
      d = dft.UKS(mol).to_gpu()


    if mf.excite == 'vlv' or mf.excite == 'pd_vlv':
      if mf.mthd == 'UKS':
        d = fock_gpu.uhf_vlv(d)
      elif mf.mthd == 'CUKS':
        d = fock_gpu.cuhf_vlv(d)
    else:
      if mf.mthd == 'UKS':
        d = fock_gpu.uhf_lvs(d)
      elif mf.mthd == 'CUKS':
        d = fock_gpu.cuhf(d)

    d.xc = mf.xc
# Construct new dnesity matrix with new occpuation pattern
    dm_d_init = d.make_rdm1(mo0, occ)

    if mf.excite == 'pd':
      import pd_gpu
      def get_occ(mo_energy=None, mo_coeff=None):
        return ro_occ
      d.get_occ = get_occ
    elif mf.excite == 'mom':
      d = scf.addons.mom_occ(d, mo0, occ)
    elif mf.excite == 'orig':
      def get_occ(mo_energy=None, mo_coeff=None):
          return occ
      d.get_occ = get_occ



# Apply mom occupation principle
    #d = scf.addons.mom_occ(d, mo0, occ)
    #pdscf
    def get_occ(mo_energy=None, mo_coeff=None):
        return occ
    d.get_occ = get_occ
    d.mo0 = mo0

# Start new SCF with new density matrix
    ed = d.scf(dm_d_init)
    d._cderi = None # delocate eri

    mo.append(d.mo_coeff)
    occa.append(d.mo_occ)
    e_d.append(ed)
    d_conv.append(d.converged)
    e_e.append(ed)

  print('xc=',mf.xc)
  print('Eg=',e_g)
  print('Ed=au,dEd=ev','SCF-converged-D')
  for i in range(len(mf.d)):
    print(i,e_d[i],27.2114*(e_d[i]-e_g),d_conv[i])
  print('Esm=au,Et=au,Es=au,dEt=ev,dEs=ev','ortho-coup','SCF-converged-T','SCF-converged-S')
  for i in range(len(mf.s)):
    dEt = 27.2114*(e_t[i]-e_g)
    oc = e_sm[i]-e_t[i]
    print(i,e_sm[i],e_t[i], e_sm[i]+oc, dEt, dEt+27.2114*2*oc, 27.2114*oc,t_conv[i],s_conv[i])
  
  #exit()
  #return

  # last ci_g
  if mf.ci_g:
    mo.append(mo0)
    occa.append(mo_occ)
    e_e.append(e_g)

  ns = len(occa)
  for i in range(ns-1):
    dip2, OV = trans_dip(mf, mol, mo[ns-1], mo[i], occa[ns-1], occa[i]) 
    osc = abs(2.0 / 3.0 * (e_e[i] - e_e[ns-1]) * dip2)    
    print('g,e', ns-1, i, '   Dipole=', mf.dips, '   Oscillator Strength=', osc, '   Overlap=', OV)

  return

class DSCF():

    def __init__(self, mol):
        self.mol = mol
        self.xc = None
        self.ci_g = True
        self.d = []
        self.s = []
        self.t = []
        #self.ro = True
        self.mthd = 'SFRO' # 'ROKS' 'UKS' 'CUKS'
        self.excite = 'pd' # 'mom': for UKS and CUKS
        self.t1_min = False # t1 use dft, not project or mom etc
        #self.mo0_iter = False # use mo in each iteration to project
        #self.mo0 = None
        self.excite_factor = 0
        self.max_cycle = 50
        self.lvs_guess = 'mo_g'
        self.dip_ov = None
        self.dips = 'Orig'
        self.h1l = False
        self.init_rodm_s_from_g = False
        self.df = True
        self.auxbasis = None
    run = run

if __name__ == '__main__':

  mol = gto.M(atom='''
 C       -4.3237417264   0.0837273697   0.0000000000
 C       -2.9800482903   0.1107880796  -0.0000000000
 H       -4.9083169485   0.9997445314  -0.0000000000
 H       -4.8762517322  -0.8496471351   0.0000000000
 H       -2.4233004597  -0.8263111708  -0.0000000000
 C       -2.1955365079   1.3394920824  -0.0000000000
 C       -0.8518430747   1.3665530392   0.0000000000
 H       -2.7522843580   2.2765913346  -0.0000000000
 H       -0.2993331643   2.2999275896   0.0000000000
 H       -0.2672677730   0.4505359139  -0.0000000000
''',
            verbose = 4,
            basis='ccpvdz')
  print('basis=',mol.basis,'nao',mol.nao)

  mf = DSCF(mol)
  mf.xc = 'pbe0' 
  mf.d = [[14,15]]
  mf.s = [[14,15],[13,15],[14,16]]
  mf.run()




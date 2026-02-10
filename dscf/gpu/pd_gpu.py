import cupy
import numba
from gpu4pyscf.scf import uhf, rohf
import jacobi_dg
import numpy
import scipy

def pd_chp(mf,fock,s):
        # Hole P'+Ph'
        tt = mf.mo_last[0][:, :mf.mol.nelec[0] + 1]
        d_a = tt@(tt.T)
        na = d_a.shape[0]
        ia = cupy.eye(na)
        pj = d_a@s
        fa = pj.T@fock[0]@pj
        e_o,c_o = pd_v(mf.mo0[0][:,:mf.mol.nelec[0]], fa) 

        # Particle P'-Pp'     
        tt = mf.mo_last[0][:, :mf.mol.nelec[0] - 1]
        d_a = tt@(tt.T)
        na = d_a.shape[0]
        ia = cupy.eye(na)
        d_v = cupy.dot(mf.mo0[0][:,mf.mol.nelec[0]:], mf.mo0[0][:,mf.mol.nelec[0]:].T)
        pj = (ia - d_a@s)
        fa = pj.T@fock[0]@pj
        e_vir,c_vir = pd_v(mf.mo0[0][:,mf.mol.nelec[0]:], fa) 

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

def pd_v_ord(mo0, fock):
        fock = mo0.T@fock@mo0
        mo_energy, coeff = cupy.linalg.eigh(fock)
        mo_coeff = mo0@coeff      
        return mo_energy, mo_coeff

def pd_v(mo0, fock):
        fock = mo0.T@fock@mo0
        A = cupy.asnumpy(fock)
        N = A.shape[0]
        A = numpy.asfortranarray(A.reshape((N,N), order='F'))  #ascontiguousarray
        c = numpy.empty((N,N), order='F')
        mo_energy = numpy.empty((N), order='F')
        jacobi_dg.diag_a(A,c,mo_energy,N)
        mo_coeff = mo0@cupy.asarray(c)
        return cupy.asarray(mo_energy), mo_coeff

def pd(mf,fock,s):
        e_a,c_a = pd_v(mf.mo0[0], fock[0])
        e_b,c_b = pd_v(mf.mo0[1], fock[1])    
        mo_energy = cupy.array((e_a,e_b)) 
        mo_coeff = cupy.array((c_a, c_b))
        return mo_energy, mo_coeff

uhf.UHF.eig = pd # pd_chp #

def pd_ro(mf,fock, s):
        return pd_v(mf.mo0, fock)

rohf.ROHF.eig = pd_ro















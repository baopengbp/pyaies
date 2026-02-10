import cupy as cp
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.dft import roks, uks

def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if dm is None:
            dm = self.make_rdm1()
        elif getattr(dm, 'mo_coeff', None) is not None:
            mo_coeff = dm.mo_coeff
            mo_occ_a = self.occa
            mo_occ_b = self.occb
            if dm.ndim == 2:
                dm = cp.repeat(dm[None]*.5, 2, axis=0)
            dm = tag_array(dm, mo_coeff=cp.asarray((mo_coeff,mo_coeff)),
                           mo_occ=cp.asarray((mo_occ_a,mo_occ_b)))
        elif dm.ndim == 2:
            dm = cp.repeat(dm[None]*.5, 2, axis=0)
        return uks.get_veff(self, mol, dm, dm_last, vhf_last, hermi)

def get_grad(self, mo_coeff, mo_occ, fock_ao):
    '''UHF Gradients'''
    if mo_occ.ndim ==2:
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
    else:
        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb

    ga = mo_coeff[:,viridxa].conj().T.dot(fock_ao.dot(mo_coeff[:,occidxa]))
    gb = mo_coeff[:,viridxb].conj().T.dot(fock_ao.dot(mo_coeff[:,occidxb]))
    return cp.hstack((ga.ravel(), gb.ravel()))

roks.ROKS.get_grad = get_grad
roks.ROKS.get_veff = get_veff

from gpu4pyscf.df import df_jk
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.dft import rks, uks, numint
from gpu4pyscf.lib import logger
import cupy
import numpy
def get_veff_df(self, mol=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        assert not self.direct_scf

        if isinstance(self, rohf.ROHF):
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = cupy.repeat(dm.mo_coeff[None], 2, axis=0)
                mo_occ = cupy.asarray([dm.mo_occ>0, dm.mo_occ==2],
                                      dtype=numpy.double)
                # new
                mo_occ_a = self.occa
                mo_occ_b = self.occb

                if dm.ndim == 2:  # RHF DM
                    dm = cupy.repeat(dm[None]*.5, 2, axis=0)
                # new
                dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=cupy.asarray((mo_occ_a,mo_occ_b)))
            elif dm.ndim == 2:  # RHF DM
                dm = cupy.repeat(dm[None]*.5, 2, axis=0)

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            t0 = logger.init_timer(self)
            rks.initialize_grids(self, mol, dm)
            ni = self._numint
            if isinstance(self, (uhf.UHF, rohf.ROHF)): # UKS
                n, exc, vxc = ni.nr_uks(mol, self.grids, self.xc, dm)
                logger.debug(self, 'nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm[0]+dm[1])
                    exc += enlc
                    vxc += vnlc
                    logger.debug(self, 'nelec with nlc grids = %s', n)
                t0 = logger.timer(self, 'vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm[0]+dm[1], hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vj = vj[0] + vj[1]
                    vxc += vj
                    vk *= hyb
                    if abs(omega) > 1e-10:
                        vklr = self.get_k(mol, dm, hermi, omega=omega)
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk
                    exc -= cupy.einsum('sij,sji->', dm, vk).real * .5
                ecoul = cupy.einsum('sij,ji->', dm, vj).real * .5

            elif isinstance(self, hf.RHF):
                n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, dm)
                logger.debug(self, 'nelec by numeric integration = %s', n)
                if self.do_nlc():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    logger.debug(self, 'nelec with nlc grids = %s', n)
                t0 = logger.timer(self, 'vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm, hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vxc += vj
                    vk *= hyb
                    if omega != 0:
                        vklr = self.get_k(mol, dm, hermi, omega=abs(omega))
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk * .5
                    exc -= cupy.einsum('ij,ji', dm, vk).real * .25
                ecoul = cupy.einsum('ij,ji', dm, vj).real * .5

            else:
                raise NotImplementedError("DF only supports R/U/RO KS.")
            t0 = logger.timer(self, 'veff', *t0)
            return tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)

        if isinstance(self, (uhf.UHF, rohf.ROHF)):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj[0] + vj[1] - vk
        elif isinstance(self, hf.RHF):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5
        else:
            raise NotImplementedError("DF only supports R/U/RO HF.")

df_jk._DFHF.get_veff = get_veff_df


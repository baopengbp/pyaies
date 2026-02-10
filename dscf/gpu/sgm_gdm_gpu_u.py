# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Square Gradient Minimization based on Geometric Direct Minimization
# ref: Hait, D.; Head-Gordon, M. J. Chem. Theory Comput. 2020, 16, 1699

import cupy
from cupyx.scipy import linalg
from pyscf import gto, scf, dft, lib

class Hybrid_Solver:
    def __init__(self, mf, diis_tol=1e-2, max_cycle_total=100):
        """
        Universal Hybrid SGM-GDM Solver (Corrected for gpu4pyscf)
        Minimizes the Squared Gradient Norm to find excited states (SGM) or converge difficult cases.
        """
        self.mf = mf
        self.mol = mf.mol
        self.diis_tol = diis_tol
        self.max_cycle_total = max_cycle_total
        
        # --- ROBUST TYPE DETECTION ---
        # 1. Standard instance check
        self.is_uhf = isinstance(mf, (scf.uhf.UHF, dft.uks.UKS))
        self.is_roks = isinstance(mf, dft.roks.ROKS)
        self.is_rohf = isinstance(mf, scf.rohf.ROHF)
        self.is_dft = isinstance(mf, (dft.rks.KohnShamDFT, dft.uks.UKS))
        
        # 2. String-based fallback for GPU-wrapped classes
        class_name = mf.__class__.__name__.upper()
        if not self.is_uhf:
            if 'UKS' in class_name or 'UHF' in class_name:
                self.is_uhf = True
        
        # Labeling
        base = "DFT" if self.is_dft else "HF"
        if self.is_uhf: ref = "U"
        elif self.is_roks: ref = "RO"
        elif self.is_rohf: ref = "RO"
        else: ref = "R"
        
        xc_label = getattr(mf, 'xc', '')
        self.method_label = f"{ref}{base}({xc_label})" if self.is_dft else f"{ref}{base}"

        # GDM Parameters
        self.conv_tol = mf.conv_tol
        self.conv_tol_grad = cupy.sqrt(mf.conv_tol)
        
        # SGM needs a reasonably conservative trust radius initially
        self.trust_radius = 0.3  
        self.min_trust = 0.001
        self.max_trust = 1.0
        
        self.grad_history = []
        self.step_history = []
        self.H_inv = None

    def _get_masks(self, mo_occ):
        # Handle cases where mo_occ is a list/tuple or a CuPy array
        if self.is_uhf:
            occ_mask_a = mo_occ[0] > 1e-6
            vir_mask_a = ~occ_mask_a
            occ_mask_b = mo_occ[1] > 1e-6
            vir_mask_b = ~occ_mask_b
            return (occ_mask_a, vir_mask_a), (occ_mask_b, vir_mask_b)
        else:
            occ_mask = mo_occ > 1e-6
            vir_mask = ~occ_mask
            return occ_mask, vir_mask

    def get_energy(self, mo_coeff, mo_occ):
        # Ensure array format for make_rdm1 in gpu4pyscf
        c_gpu = mo_coeff
        o_gpu = mo_occ
        if self.is_uhf:
            if isinstance(mo_coeff, (tuple, list)): c_gpu = cupy.stack(mo_coeff)
            if isinstance(mo_occ, (tuple, list)): o_gpu = cupy.stack(mo_occ)
            
        dm = self.mf.make_rdm1(c_gpu, o_gpu)
        return self.mf.energy_tot(dm=dm)

    def get_fock(self, mo_coeff, mo_occ):
        """
        Computes Fock matrix. 
        CRITICAL FIX: gpu4pyscf requires 'mo_coeff' attached to 'mf' to be a CuPy array 
        of shape (2, N, N) for UKS, not a tuple.
        """
        mo_coeff_gpu = mo_coeff
        mo_occ_gpu = mo_occ
        
        if self.is_uhf:
            if isinstance(mo_coeff, (tuple, list)):
                mo_coeff_gpu = cupy.stack(mo_coeff)
            if isinstance(mo_occ, (tuple, list)):
                mo_occ_gpu = cupy.stack(mo_occ)

        # Update mf object with GPU-friendly arrays
        self.mf.mo_coeff = mo_coeff_gpu
        self.mf.mo_occ = mo_occ_gpu
        
        dm = self.mf.make_rdm1(mo_coeff_gpu, mo_occ_gpu)
        return self.mf.get_fock(dm=dm)

    # --- Geometry Ops ---
    def pseudo_canonicalize_block(self, fock_mo, occ_mask, vir_mask):
        Foo = fock_mo[cupy.ix_(occ_mask, occ_mask)]
        Fvv = fock_mo[cupy.ix_(vir_mask, vir_mask)]
        
        eig_occ, U_occ = cupy.linalg.eigh(Foo)
        eig_vir, U_vir = cupy.linalg.eigh(Fvv)
        
        n_tot = fock_mo.shape[0]
        U = cupy.zeros((n_tot, n_tot))
        
        U[cupy.ix_(occ_mask, occ_mask)] = U_occ
        U[cupy.ix_(vir_mask, vir_mask)] = U_vir
        
        all_eps = cupy.zeros(n_tot)
        all_eps[occ_mask] = eig_occ
        all_eps[vir_mask] = eig_vir
        
        return U, all_eps

    def pseudo_canonicalize(self, fock_mo, masks):
        if self.is_uhf:
            (occ_a, vir_a), (occ_b, vir_b) = masks
            U_a, eps_a = self.pseudo_canonicalize_block(fock_mo[0], occ_a, vir_a)
            U_b, eps_b = self.pseudo_canonicalize_block(fock_mo[1], occ_b, vir_b)
            return (U_a, U_b), (eps_a, eps_b)
        else:
            occ_mask, vir_mask = masks
            return self.pseudo_canonicalize_block(fock_mo, occ_mask, vir_mask)

    def apply_rotation(self, mo_coeff, U_rot):
        if self.is_uhf:
            # Returns a tuple, handled correctly by get_fock conversion
            return (mo_coeff[0] @ U_rot[0], mo_coeff[1] @ U_rot[1])
        return mo_coeff @ U_rot

    def parallel_transport_list(self, history_list, U_rot, masks):
        new_list = []
        for item in history_list:
            if self.is_uhf:
                (occ_a, vir_a), (occ_b, vir_b) = masks
                Ua_oo = U_rot[0][cupy.ix_(occ_a, occ_a)]
                Ua_vv = U_rot[0][cupy.ix_(vir_a, vir_a)]
                Ub_oo = U_rot[1][cupy.ix_(occ_b, occ_b)]
                Ub_vv = U_rot[1][cupy.ix_(vir_b, vir_b)]
                new_list.append((Ua_oo.T @ item[0] @ Ua_vv, Ub_oo.T @ item[1] @ Ub_vv))
            else:
                occ_mask, vir_mask = masks
                U_oo = U_rot[cupy.ix_(occ_mask, occ_mask)]
                U_vv = U_rot[cupy.ix_(vir_mask, vir_mask)]
                new_list.append(U_oo.T @ item @ U_vv)
        return new_list

    def get_exp_map(self, step_ov, masks, shapes):
        if self.is_uhf:
            def _build(step, occ_mask, vir_mask, nmo):
                gen = cupy.zeros((nmo, nmo))
                gen[cupy.ix_(occ_mask, vir_mask)] = -step
                gen[cupy.ix_(vir_mask, occ_mask)] = step.T
                return linalg.expm(gen)
            
            (occ_a, vir_a), (occ_b, vir_b) = masks
            return (_build(step_ov[0], occ_a, vir_a, shapes[0]), 
                    _build(step_ov[1], occ_b, vir_b, shapes[1]))
        else:
            occ_mask, vir_mask = masks
            nmo = shapes
            gen = cupy.zeros((nmo, nmo))
            gen[cupy.ix_(occ_mask, vir_mask)] = -step_ov
            gen[cupy.ix_(vir_mask, occ_mask)] = step_ov.T
            return linalg.expm(gen)

    # --- BFGS Utils ---
    def pack_vector(self, data):
        if self.is_uhf:
            return cupy.concatenate((data[0].reshape(-1), data[1].reshape(-1)))
        return data.reshape(-1)

    def unpack_vector(self, vec, masks, shapes):
        if self.is_uhf:
            (occ_a, vir_a), (occ_b, vir_b) = masks
            nocc_a = int(cupy.sum(occ_a)); nvir_a = int(cupy.sum(vir_a))
            nocc_b = int(cupy.sum(occ_b)); nvir_b = int(cupy.sum(vir_b))
            size_a = nocc_a * nvir_a
            return (vec[:size_a].reshape(nocc_a, nvir_a), 
                    vec[size_a:].reshape(nocc_b, nvir_b))
        else:
            occ_mask, vir_mask = masks
            nocc = int(cupy.sum(occ_mask))
            nvir = int(cupy.sum(vir_mask))
            return vec.reshape(nocc, nvir)

    def compute_preconditioner(self, eps, masks):
        if self.is_uhf:
            def _get_diag(e, occ_mask, vir_mask):
                e_occ = e[occ_mask]
                e_vir = e[vir_mask]
                B = (e_vir.reshape(1,-1) - e_occ.reshape(-1,1))
                B = cupy.maximum(cupy.abs(B), 0.1) 
                return 1.0 / B
            
            (occ_a, vir_a), (occ_b, vir_b) = masks
            return (_get_diag(eps[0], occ_a, vir_a), _get_diag(eps[1], occ_b, vir_b))
        else:
            occ_mask, vir_mask = masks
            e_occ = eps[occ_mask]
            e_vir = eps[vir_mask]
            B = (e_vir.reshape(1,-1) - e_occ.reshape(-1,1))
            B = cupy.maximum(cupy.abs(B), 0.1)
            return 1.0 / B

    def apply_precon(self, data, precon, inverse=False):
        if self.is_uhf:
            op = (lambda x, y: x / y) if inverse else (lambda x, y: x * y)
            return (op(data[0], precon[0]), op(data[1], precon[1]))
        return (data / precon) if inverse else (data * precon)

    def bfgs_update(self, s, y, H_inv):
        s = s.reshape(-1, 1); y = y.reshape(-1, 1)
        sy = cupy.dot(s.T, y)
        if sy < 1e-12: return H_inv 
        rho = 1.0 / sy
        I = cupy.eye(len(s))
        return (I - rho*s@y.T) @ H_inv @ (I - rho*y@s.T) + rho*s@s.T

    # --- Gradient Helper ---
    def get_raw_gradient(self, mo_coeff, mo_occ):
        masks = self._get_masks(mo_occ)
        fock_ao = self.get_fock(mo_coeff, mo_occ)
        
        if self.is_uhf:
            # fock_ao is (2, N, N) array from gpu4pyscf
            # mo_coeff might be tuple or (2, N, N) array
            # We use indexing [0]/[1] which works for both
            
            fock_mo_a = mo_coeff[0].T @ fock_ao[0] @ mo_coeff[0]
            fock_mo_b = mo_coeff[1].T @ fock_ao[1] @ mo_coeff[1]
            
            (occ_a, vir_a), (occ_b, vir_b) = masks
            grad_a = fock_mo_a[cupy.ix_(occ_a, vir_a)]
            grad_b = fock_mo_b[cupy.ix_(occ_b, vir_b)]
            return (grad_a, grad_b)
        else:
            fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
            occ_mask, vir_mask = masks
            return fock_mo[cupy.ix_(occ_mask, vir_mask)]

    def get_grad_norm_sq(self, grad_data):
        if self.is_uhf:
            return cupy.linalg.norm(grad_data[0])**2 + cupy.linalg.norm(grad_data[1])**2
        return cupy.linalg.norm(grad_data)**2

    # --- KERNEL ---
    def kernel(self):
        print(f"\n Method: {self.method_label} | Hybrid SGM-GDM Solver")
        print(f" Minimizing Square Gradient Norm to find stationary point (Saddle/Min).")
        print("-" * 96)
        print(f"{'Iter':<5} {'Phase':<6} {'Total Energy':<16} {'|Grad|^2':<12} {'Grad RMS':<12} {'Trust':<10}")
        print("-" * 96)

        self.mf.max_cycle = self.max_cycle_total
        
        # Init guess checks
        if self.mf.mo_coeff is None:
            if self.is_dft and self.mf.grids.coords is None: self.mf.grids.build()
            dm = self.mf.get_init_guess()
            fock = self.mf.get_fock(dm=dm)
            s1e = self.mf.get_ovlp()
            mo_energy, mo_coeff = self.mf.eig(fock, s1e)
            mo_occ = self.mf.get_occ(mo_energy, mo_coeff)
            self.mf.mo_coeff = mo_coeff
            self.mf.mo_occ = mo_occ
        
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        
        # Determine shapes
        if self.is_uhf: 
            shapes = (mo_coeff[0].shape[0], mo_coeff[1].shape[0])
        else: 
            shapes = mo_coeff.shape[0]
        
        # Initial status
        g0 = self.get_raw_gradient(mo_coeff, mo_occ)
        obj_curr = self.get_grad_norm_sq(g0)
        
        curr_cycle = 0
        self.H_inv = None
        
        while curr_cycle < self.max_cycle_total:
            masks = self._get_masks(mo_occ)
            energy_real = self.get_energy(mo_coeff, mo_occ)

            # 1. Rotate & Parallel Transport
            fock_ao = self.get_fock(mo_coeff, mo_occ)
            if self.is_uhf:
                fock_mo = (mo_coeff[0].T@fock_ao[0]@mo_coeff[0], mo_coeff[1].T@fock_ao[1]@mo_coeff[1])
            else:
                fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
                
            U_rot, eps = self.pseudo_canonicalize(fock_mo, masks)
            mo_coeff = self.apply_rotation(mo_coeff, U_rot)
            
            # Transport
            self.grad_history = self.parallel_transport_list(self.grad_history, U_rot, masks)
            self.step_history = self.parallel_transport_list(self.step_history, U_rot, masks)
            
            # Recalculate g0
            g0 = self.get_raw_gradient(mo_coeff, mo_occ)
            obj_curr = self.get_grad_norm_sq(g0)
            grad_norm0 = cupy.sqrt(obj_curr)

            if grad_norm0 < self.conv_tol_grad:
                print(f"{curr_cycle:<5} {'SGM':<6} {energy_real:<16.12f} {obj_curr:<12.2e} {grad_norm0:<12.2e} {'Done':<10}")
                print("\nConverged!")
                self.mf.mo_coeff = mo_coeff
                self.mf.e_tot = energy_real
                return energy_real

            # 2. Gradient of Objective (Finite Difference)
            lam = 1e-6
            
            if self.is_uhf:
                perturbation = (g0[0] * lam, g0[1] * lam)
                neg_pert = (-perturbation[0], -perturbation[1])
            else:
                perturbation = g0 * lam
                neg_pert = -perturbation

            mo_plus = self.apply_rotation(mo_coeff, self.get_exp_map(perturbation, masks, shapes))
            g_plus = self.get_raw_gradient(mo_plus, mo_occ)
            
            mo_minus = self.apply_rotation(mo_coeff, self.get_exp_map(neg_pert, masks, shapes))
            g_minus = self.get_raw_gradient(mo_minus, mo_occ)
            
            if self.is_uhf:
                grad_data = ((g_plus[0] - g_minus[0])/(2*lam), (g_plus[1] - g_minus[1])/(2*lam))
            else:
                grad_data = (g_plus - g_minus) / (2*lam)

            # 3. Precon & BFGS
            precon = self.compute_preconditioner(eps, masks)
            
            grad_ewc = self.apply_precon(grad_data, precon, inverse=False)
            grad_vec = self.pack_vector(grad_ewc)
            
            if self.H_inv is None:
                dim = len(grad_vec)
                self.H_inv = cupy.eye(dim) 
            
            if len(self.grad_history) > 0:
                g_prev_vec = self.pack_vector(self.apply_precon(self.grad_history[-1], precon, inverse=False))
                s_prev_vec = self.pack_vector(self.apply_precon(self.step_history[-1], precon, inverse=True))
                self.H_inv = self.bfgs_update(s_prev_vec, grad_vec - g_prev_vec, self.H_inv)

            # 4. Step
            step_ewc_vec = -self.H_inv @ grad_vec
            step_ewc_data = self.unpack_vector(step_ewc_vec, masks, shapes)
            step_ov = self.apply_precon(step_ewc_data, precon, inverse=False)
            
            if self.is_uhf: step_norm = cupy.sqrt(cupy.linalg.norm(step_ov[0])**2 + cupy.linalg.norm(step_ov[1])**2)
            else: step_norm = cupy.linalg.norm(step_ov)
            
            scale = 1.0
            if step_norm > self.trust_radius:
                scale = self.trust_radius / step_norm
                if self.is_uhf: step_ov = (step_ov[0]*scale, step_ov[1]*scale)
                else: step_ov *= scale

            # 5. Line Search
            alpha = 1.0; accepted = False
            actual_step = None
            
            for i in range(5):
                if self.is_uhf: curr = (step_ov[0]*alpha, step_ov[1]*alpha)
                else: curr = step_ov * alpha
                
                mo_trial = self.apply_rotation(mo_coeff, self.get_exp_map(curr, masks, shapes))
                
                g_trial = self.get_raw_gradient(mo_trial, mo_occ)
                sq_grad_trial = self.get_grad_norm_sq(g_trial)
                
                if sq_grad_trial < obj_curr:
                    mo_coeff = mo_trial
                    accepted = True
                    actual_step = curr
                    
                    if scale < 1.0 and i == 0: 
                        self.trust_radius = min(self.max_trust, self.trust_radius * 1.5)
                    elif i > 1: 
                        self.trust_radius = max(self.min_trust, self.trust_radius * 0.8)
                    break
                alpha *= 0.5
            
            if not accepted:
                self.H_inv = None; self.grad_history = []; self.step_history = []
                self.trust_radius = max(self.min_trust, self.trust_radius * 0.5)
                
                sd_data = self.apply_precon(self.apply_precon(grad_data, precon), precon)
                if self.is_uhf: actual_step = (-sd_data[0]*0.1, -sd_data[1]*0.1)
                else: actual_step = -sd_data * 0.1
                
                mo_coeff = self.apply_rotation(mo_coeff, self.get_exp_map(actual_step, masks, shapes))

            print(f"{curr_cycle:<5} {'SGM':<6} {energy_real:<16.12f} {obj_curr:<12.2e} {grad_norm0:<12.2e} {self.trust_radius:<10.2e}")
            
            self.grad_history.append(grad_data)
            self.step_history.append(actual_step)
            
            if len(self.grad_history) > 20: 
                self.grad_history.pop(0)
                self.step_history.pop(0)
            
            curr_cycle += 1
            
        return energy_real

if __name__ == "__main__":
    print("\n--- TEST: UKS Oxygen (B3LYP) ---")
    mol = gto.M(atom='''
C    0.00000000            0.00000000           -0.60298508
O    0.00000000            0.00000000            0.60539399
H    0.00000000            0.93467313           -1.18217476
H    0.00000000           -0.93467313           -1.18217476
''',
            basis='def2-svp'
    )
    mf = dft.UKS(mol).density_fit().to_gpu()
    mf.xc = 'pbe0'
    mf.verbose = 0
    mf.kernel()

    print(f"Ground State Energy: {mf.e_tot}")

    if mf.mo_occ[0][mol.nelec[0]-1] > 0.5: # Alpha channel
        homo_idx = mol.nelec[0]-1
        lumo_idx = homo_idx + 1
        print(f"Swapping Alpha orbitals {homo_idx} and {lumo_idx}")
        
        mo_new = mf.mo_coeff.copy()
        # Ensure we are modifying a copy and handling array vs tuple
        if isinstance(mo_new, (tuple, list)):
            mo_new = [c.copy() for c in mo_new]
            tmp = mo_new[0][:, homo_idx].copy()
            mo_new[0][:, homo_idx] = mo_new[0][:, lumo_idx]
            mo_new[0][:, lumo_idx] = tmp
            mo_new = tuple(mo_new)
        else:
            tmp = mo_new[0][:, homo_idx].copy()
            mo_new[0][:, homo_idx] = mo_new[0][:, lumo_idx]
            mo_new[0][:, lumo_idx] = tmp
        
        mf.mo_coeff = mo_new

    print("\nStarting SGM Solver on Excited State Guess...")
    Hybrid_Solver(mf, max_cycle_total=100).kernel()




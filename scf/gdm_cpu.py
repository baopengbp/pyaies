#
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Geometric Direct Minimization
# ref: Van Voorhis, T.; Head-Gordon, M. Mol. Phys. 2002, 100, 1713−1721.

import numpy as np
from scipy import linalg
from pyscf import gto, scf, dft, lib
from pyscf.scf import rohf 

class Hybrid_Solver:
    def __init__(self, mf, diis_tol=1e-2, max_cycle_total=100):
        """
        Universal Hybrid DIIS-GDM Solver.
        Optimized to minimize vhf/get_veff calls.
        """
        self.mf = mf
        self.mol = mf.mol
        self.diis_tol = diis_tol
        self.max_cycle_total = max_cycle_total
        
        # Type detection
        self.is_uhf = isinstance(mf, scf.uhf.UHF)
        self.is_roks = isinstance(mf, dft.roks.ROKS)
        self.is_rohf = isinstance(mf, scf.rohf.ROHF)
        self.is_dft = isinstance(mf, dft.rks.KohnShamDFT)
        self.is_restricted_open = self.is_roks or self.is_rohf

        # Label
        base = "DFT" if self.is_dft else "HF"
        if self.is_uhf: ref = "U"
        elif self.is_restricted_open: ref = "RO"
        else: ref = "R"
        self.method_label = f"{ref}{base}"
        if self.is_dft: self.method_label += f"({mf.xc})"

        # GDM Parameters
        self.conv_tol = mf.conv_tol
        self.conv_tol_grad = np.sqrt(mf.conv_tol)
        self.trust_radius = 0.1
        self.min_trust = 0.001
        self.max_trust = 0.5
        
        self.grad_history = []
        self.step_history = []
        self.H_inv = None

    def _get_masks(self, mo_occ):
        if self.is_uhf:
            occ_mask_a = mo_occ[0] > 1e-6
            vir_mask_a = ~occ_mask_a
            occ_mask_b = mo_occ[1] > 1e-6
            vir_mask_b = ~occ_mask_b
            return (occ_mask_a, vir_mask_a), (occ_mask_b, vir_mask_b)
        elif self.is_restricted_open:
            closed_mask = mo_occ > 1.5
            open_mask = (mo_occ > 0.5) & (mo_occ < 1.5)
            vir_mask = mo_occ < 0.5
            return (closed_mask, open_mask, vir_mask)
        else:
            occ_mask = mo_occ > 1e-6
            vir_mask = ~occ_mask
            return occ_mask, vir_mask

    def eval_physics(self, mo_coeff, mo_occ):
        #print('sss')
        """
        Compute Energy and Fock matrix simultaneously to avoid recomputing vhf.
        This is the most expensive step (O(N^4) or Grid).
        """
        # 1. Build Density Matrix
        dm = self.mf.make_rdm1(mo_coeff, mo_occ)
        
        # 2. Compute V_eff (Coulomb + XC/Exchange) - Expensive!
        # This is the only place get_veff should be called per trial step.
        vhf = self.mf.get_veff(self.mol, dm)
        
        # 3. Get H_core (usually cached)
        h1e = self.mf.get_hcore()

        # 4. Compute Energy using precomputed vhf
        e_tot = self.mf.energy_tot(dm=dm, h1e=h1e, vhf=vhf)
        
        # 5. Compute Fock using precomputed vhf
        # For ROKS, this correctly assembles the unified operator
        fock = self.mf.get_fock(dm=dm, h1e=h1e, vhf=vhf)
        
        return e_tot, fock

    # --- Geometry Ops ---
    def _canonicalize_generic(self, fock, masks_list):
        nmo = fock.shape[0]
        U = np.eye(nmo)
        all_eps = np.zeros(nmo)
        
        for mask in masks_list:
            if np.sum(mask) > 0:
                sub_f = fock[np.ix_(mask, mask)]
                w, v = linalg.eigh(sub_f)
                U[np.ix_(mask, mask)] = v
                all_eps[mask] = w
                
        return U, all_eps, U.T @ fock @ U

    def pseudo_canonicalize(self, fock_mo, masks):
        if self.is_uhf:
            (masks_a, masks_b) = masks
            Ua, ea, Fa = self._canonicalize_generic(fock_mo[0], masks_a)
            Ub, eb, Fb = self._canonicalize_generic(fock_mo[1], masks_b)
            return (Ua, Ub), (ea, eb), (Fa, Fb)
        else:
            return self._canonicalize_generic(fock_mo, masks)

    def apply_rotation(self, mo_coeff, U_rot):
        if self.is_uhf:
            return (mo_coeff[0] @ U_rot[0], mo_coeff[1] @ U_rot[1])
        return mo_coeff @ U_rot

    def parallel_transport_list(self, history_list, U_rot, masks):
        new_list = []
        for item in history_list:
            if self.is_uhf:
                (oa, va), (ob, vb) = masks
                Ua_oo = U_rot[0][np.ix_(oa, oa)]; Ua_vv = U_rot[0][np.ix_(va, va)]
                ga = Ua_oo.T @ item[0] @ Ua_vv
                Ub_oo = U_rot[1][np.ix_(ob, ob)]; Ub_vv = U_rot[1][np.ix_(vb, vb)]
                gb = Ub_oo.T @ item[1] @ Ub_vv
                new_list.append((ga, gb))
            elif self.is_restricted_open:
                mc, mo, mv = masks
                Uc = U_rot[np.ix_(mc, mc)]
                Uo = U_rot[np.ix_(mo, mo)]
                Uv = U_rot[np.ix_(mv, mv)]
                g_co, g_cv, g_ov = item
                new_list.append((Uc.T@g_co@Uo, Uc.T@g_cv@Uv, Uo.T@g_ov@Uv))
            else:
                occ_mask, vir_mask = masks
                U_oo = U_rot[np.ix_(occ_mask, occ_mask)]
                U_vv = U_rot[np.ix_(vir_mask, vir_mask)]
                new_list.append(U_oo.T @ item @ U_vv)
        return new_list

    def get_exp_map(self, step_data, masks, shapes):
        """
        Geodesic step via SVD (Eq 13). 
        Corrected signs to match descent direction.
        """
        def _build_geodesic_unit(step, mask_row, mask_col, nmo):
            if step.size == 0: return np.eye(nmo)
            u, s, vh = linalg.svd(step, full_matrices=False)
            v = vh.T
            cos_s = np.cos(s); sin_s = np.sin(s)
            
            idx_row = np.where(mask_row)[0]
            idx_col = np.where(mask_col)[0]
            
            # Corrected signs for descent:
            # Block Row-Col (Occ-Vir): -U sin V^T
            # Block Col-Row (Vir-Occ): +V sin U^T
            d_rr = -u @ (np.diag(1.0 - cos_s) @ u.T)
            d_cc = -v @ (np.diag(1.0 - cos_s) @ v.T)
            d_rc = -u @ (np.diag(sin_s) @ vh) 
            d_cr =  v @ (np.diag(sin_s) @ u.T)

            U_rot = np.eye(nmo)
            U_rot[np.ix_(idx_row, idx_row)] += d_rr
            U_rot[np.ix_(idx_col, idx_col)] += d_cc
            U_rot[np.ix_(idx_row, idx_col)] += d_rc
            U_rot[np.ix_(idx_col, idx_row)] += d_cr
            return U_rot

        if self.is_uhf:
            (oa, va), (ob, vb) = masks
            return (_build_geodesic_unit(step_data[0], oa, va, shapes[0]),
                    _build_geodesic_unit(step_data[1], ob, vb, shapes[1]))
        elif self.is_restricted_open:
            mc, mo, mv = masks
            nmo = shapes
            U_co = _build_geodesic_unit(step_data[0], mc, mo, nmo)
            U_cv = _build_geodesic_unit(step_data[1], mc, mv, nmo)
            U_ov = _build_geodesic_unit(step_data[2], mo, mv, nmo)
            return U_co @ U_cv @ U_ov
        else:
            occ_mask, vir_mask = masks
            return _build_geodesic_unit(step_data, occ_mask, vir_mask, shapes)

    # --- BFGS Utils ---
    def pack_vector(self, data):
        if self.is_uhf: return np.concatenate((data[0].reshape(-1), data[1].reshape(-1)))
        elif self.is_restricted_open: return np.concatenate([x.reshape(-1) for x in data])
        else: return data.reshape(-1)

    def unpack_vector(self, vec, masks, shapes):
        if self.is_uhf:
            (oa, va), (ob, vb) = masks
            na_o, na_v = np.sum(oa), np.sum(va)
            nb_o, nb_v = np.sum(ob), np.sum(vb)
            split = na_o * na_v
            return (vec[:split].reshape(na_o, na_v), vec[split:].reshape(nb_o, nb_v))
        elif self.is_restricted_open:
            mc, mo, mv = masks
            nc, no, nv = np.sum(mc), np.sum(mo), np.sum(mv)
            s1 = nc*no; s2 = s1 + nc*nv
            return (vec[:s1].reshape(nc, no), vec[s1:s2].reshape(nc, nv), vec[s2:].reshape(no, nv))
        else:
            occ_mask, vir_mask = masks
            nocc = np.sum(occ_mask); nvir = np.sum(vir_mask)
            return vec.reshape(nocc, nvir)

    def compute_preconditioner(self, eps, masks, shift):
        def _make_diag(e_virt, e_occ, shift_val):
            B = 2.0 * (e_virt.reshape(1,-1) - e_occ.reshape(-1,1)) + shift_val
            return 1.0/np.sqrt(np.maximum(B, 0.1))

        if self.is_uhf:
            (oa, va), (ob, vb) = masks
            return (_make_diag(eps[0][va], eps[0][oa], shift),
                    _make_diag(eps[1][vb], eps[1][ob], shift))
        elif self.is_restricted_open:
            mc, mo, mv = masks
            e_c = eps[mc]; e_o = eps[mo]; e_v = eps[mv]
            return (_make_diag(e_o, e_c, shift), _make_diag(e_v, e_c, shift), _make_diag(e_v, e_o, shift))
        else:
            occ_mask, vir_mask = masks
            return _make_diag(eps[vir_mask], eps[occ_mask], shift)

    def apply_precon(self, data, precon, inverse=False):
        op = (lambda x, y: x / y) if inverse else (lambda x, y: x * y)
        if self.is_uhf: return (op(data[0], precon[0]), op(data[1], precon[1]))
        elif self.is_restricted_open: return (op(data[0], precon[0]), op(data[1], precon[1]), op(data[2], precon[2]))
        else: return op(data, precon)

    def get_lbfgs_step(self, grad_vec, s_history, y_history, precon, masks):
        """
        Calculate step direction using L-BFGS Two-Loop Recursion.
        Complexity: O(m*N), Memory: O(m*N). Replaces O(N^2) H_inv matrix.
        """
        m = len(s_history)
        if m == 0:
            return -grad_vec

        # 1. Prepare vectors aligned to current Preconditioner
        # s_vec = s_raw / precon (Inverse=True)
        # y_vec = y_raw * precon (Inverse=False)
        # This ensures we work in the "Energy Weighted" Euclidean space
        s_vecs = []
        y_vecs = []
        rhos = []

        # Process history (Recent is at the end)
        for i in range(m):
            sv = self.pack_vector(self.apply_precon(s_history[i], precon, inverse=True))
            yv = self.pack_vector(self.apply_precon(y_history[i], precon, inverse=False))
            
            sy = np.dot(sv, yv)
            if sy > 1e-12: # Numerical stability check
                s_vecs.append(sv)
                y_vecs.append(yv)
                rhos.append(1.0 / sy)
        
        m_eff = len(s_vecs)
        if m_eff == 0: return -grad_vec

        # 2. Backward Loop
        q = grad_vec.copy()
        alphas = np.zeros(m_eff)
        
        for i in range(m_eff - 1, -1, -1):
            alphas[i] = rhos[i] * np.dot(s_vecs[i], q)
            q -= alphas[i] * y_vecs[i]

        # 3. Scaling (H0 approx)
        # gamma = (s_last . y_last) / (y_last . y_last)
        s_last = s_vecs[-1]
        y_last = y_vecs[-1]
        gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        r = q * gamma

        # 4. Forward Loop
        for i in range(m_eff):
            beta = rhos[i] * np.dot(y_vecs[i], r)
            r += s_vecs[i] * (alphas[i] - beta)

        return -r # Direction is -H * g

    # --- KERNEL ---
    def kernel(self):
        print(f"Method: {self.method_label} | Solver: L-BFGS (Optimized)")
        print(f"Geometric Direct Minimization")
        print("-" * 88)
        print(f"{'Iter':<5} {'Phase':<6} {'Energy':<16} {'Delta E':<12} {'Grad RMS':<12} {'Trust/Info':<10}")
        print("-" * 88)

        self.mf.max_cycle = self.max_cycle_total
        mol = self.mf.mol
        
        if self.mf.mo_coeff is None:
            if self.is_dft and self.mf.grids.coords is None: self.mf.grids.build()
            dm = self.mf.get_init_guess()

            h1e = self.mf.get_hcore(mol)
            vhf = self.mf.get_veff(mol, dm)
            e_init = self.mf.energy_tot(dm, h1e, vhf)
            print('init E= ', e_init)

            s1e = self.mf.get_ovlp()
            fock_ao = self.mf.get_fock(h1e, s1e, vhf, dm)

            mo_energy, mo_coeff = self.mf.eig(fock_ao, s1e)
            mo_occ = self.mf.get_occ(mo_energy, mo_coeff)
            self.mf.mo_coeff = mo_coeff
            self.mf.mo_occ = mo_occ
        
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        masks = self._get_masks(mo_occ)
        if self.is_uhf: shapes = (mo_coeff[0].shape[0], mo_coeff[1].shape[0])
        else: shapes = mo_coeff.shape[0]

        # Init Physics
        e_curr, fock_ao = self.eval_physics(mo_coeff, mo_occ)
        e_last = e_curr
        
        # New History Containers for L-BFGS
        self.s_history = [] 
        self.y_history = []
        last_grad_data = None # To compute y = g_new - g_old
        
        curr_cycle = 0
        
        while curr_cycle < self.max_cycle_total:
            masks = self._get_masks(mo_occ)

            # 1. Rotate & Parallel Transport
            if self.is_uhf:
                fock_mo = (mo_coeff[0].T@fock_ao[0]@mo_coeff[0], mo_coeff[1].T@fock_ao[1]@mo_coeff[1])
            else:
                fock_mo = mo_coeff.T @ fock_ao @ mo_coeff
                
            U_rot, eps, fock_canon = self.pseudo_canonicalize(fock_mo, masks)
            mo_coeff = self.apply_rotation(mo_coeff, U_rot)
            
            # Transport History Matrices
            self.s_history = self.parallel_transport_list(self.s_history, U_rot, masks)
            self.y_history = self.parallel_transport_list(self.y_history, U_rot, masks)
            if last_grad_data is not None:
                # Need to transport the reference gradient too to compute y correctly
                last_grad_data = self.parallel_transport_list([last_grad_data], U_rot, masks)[0]
            
            # 2. Gradient Calculation
            if self.is_uhf:
                (oa, va), (ob, vb) = masks
                grad_data = (fock_canon[0][np.ix_(oa, va)], fock_canon[1][np.ix_(ob, vb)])
                grad_norm = np.sqrt(np.linalg.norm(grad_data[0])**2 + np.linalg.norm(grad_data[1])**2)
            elif self.is_restricted_open:
                mc, mo, mv = masks
                g_co = fock_canon[np.ix_(mc, mo)]; g_cv = fock_canon[np.ix_(mc, mv)]; g_ov = fock_canon[np.ix_(mo, mv)]
                grad_data = (g_co, g_cv, g_ov)
                grad_norm = np.sqrt(np.linalg.norm(g_co)**2 + np.linalg.norm(g_cv)**2 + np.linalg.norm(g_ov)**2)
            else:
                occ_mask, vir_mask = masks
                grad_data = fock_canon[np.ix_(occ_mask, vir_mask)]
                grad_norm = np.linalg.norm(grad_data)
                
            delta_e = e_curr - e_last
            
            if grad_norm < self.conv_tol_grad and abs(delta_e) < self.conv_tol and curr_cycle > 0:
                print(f"{curr_cycle:<5} {'GDM':<6} {e_curr:<16.8f} {delta_e:<12.2e} {grad_norm:<12.2e} {'Done':<10}")
                self.mf.mo_coeff = mo_coeff; self.mf.e_tot = e_curr
                return e_curr
            
            # Update History with previous step's y = g_curr - g_prev
            if last_grad_data is not None:
                # Compute difference in raw matrix form
                if self.is_uhf:
                    y_raw = (grad_data[0] - last_grad_data[0], grad_data[1] - last_grad_data[1])
                elif self.is_restricted_open:
                    y_raw = (grad_data[0]-last_grad_data[0], grad_data[1]-last_grad_data[1], grad_data[2]-last_grad_data[2])
                else:
                    y_raw = grad_data - last_grad_data
                
                self.y_history.append(y_raw)
                # s_history was already appended at end of previous cycle
                
                if len(self.s_history) > 20: 
                    self.s_history.pop(0)
                    self.y_history.pop(0)

            # 3. Precon & Step
            gap_shift = 0.0 if delta_e < 0 else 0.1
            precon = self.compute_preconditioner(eps, masks, gap_shift)
            
            grad_ewc = self.apply_precon(grad_data, precon, inverse=False)
            grad_vec = self.pack_vector(grad_ewc)
            
            # --- L-BFGS Step (O(mN)) ---
            step_ewc_vec = self.get_lbfgs_step(grad_vec, self.s_history, self.y_history, precon, masks)
            
            step_ewc_data = self.unpack_vector(step_ewc_vec, masks, shapes)
            step_data = self.apply_precon(step_ewc_data, precon, inverse=False)
            
            step_norm = np.linalg.norm(self.pack_vector(step_data))
            scale = 1.0
            if step_norm > self.trust_radius:
                scale = self.trust_radius / step_norm
                if self.is_uhf: step_data = (step_data[0]*scale, step_data[1]*scale)
                elif self.is_restricted_open: step_data = (step_data[0]*scale, step_data[1]*scale, step_data[2]*scale)
                else: step_data *= scale

            # 4. Backtrack
            alpha = 1.0; accepted = False
            next_e = e_curr; next_fock = None; next_mo = None
            
            for i in range(5):
                if self.is_uhf: curr = (step_data[0]*alpha, step_data[1]*alpha)
                elif self.is_restricted_open: curr = (step_data[0]*alpha, step_data[1]*alpha, step_data[2]*alpha)
                else: curr = step_data * alpha
                
                mo_trial = self.apply_rotation(mo_coeff, self.get_exp_map(curr, masks, shapes))
                e_trial, fock_trial = self.eval_physics(mo_trial, mo_occ)
                
                if e_trial < e_curr + 1e-8:
                    next_mo = mo_trial; next_e = e_trial; next_fock = fock_trial
                    accepted = True; actual_step = curr
                    if scale < 1.0 and i == 0: self.trust_radius = min(self.max_trust, self.trust_radius * 1.5)
                    elif i == 0: self.trust_radius = min(self.max_trust, self.trust_radius * 1.2)
                    break
                alpha *= 0.5
            
            if not accepted:
                self.s_history = []; self.y_history = [] # Reset L-BFGS
                self.trust_radius = max(self.min_trust, self.trust_radius * 0.5)
                
                sd_data = self.apply_precon(self.apply_precon(grad_data, precon), precon)
                if self.is_uhf: actual_step = (-sd_data[0]*0.2, -sd_data[1]*0.2)
                elif self.is_restricted_open: actual_step = (-sd_data[0]*0.2, -sd_data[1]*0.2, -sd_data[2]*0.2)
                else: actual_step = -sd_data * 0.2
                
                mo_trial = self.apply_rotation(mo_coeff, self.get_exp_map(actual_step, masks, shapes))
                next_e, next_fock = self.eval_physics(mo_trial, mo_occ)
                next_mo = mo_trial

            print(f"{curr_cycle:<5} {'GDM':<6} {e_curr:<16.8f} {delta_e:<12.2e} {grad_norm:<12.2e} {self.trust_radius:<10.2e}")
            
            # Commit
            e_last = e_curr; e_curr = next_e
            mo_coeff = next_mo; fock_ao = next_fock
            
            # Update History
            last_grad_data = grad_data # Store current grad for next iter's y calculation
            self.s_history.append(actual_step) # Store actual step taken for next iter's s
            
            curr_cycle += 1
            
        return e_curr

if __name__ == "__main__":
    mol = gto.M(atom='''
C    0.00000000            0.00000000           -0.60298508
O    0.00000000            0.00000000            0.60539399
H    0.00000000            0.93467313           -1.18217476
H    0.00000000           -0.93467313           -1.18217476
''',
            basis='def2-svp'
    )

    # 1. RKS
    print("\n--- TEST: RKS ---")
    mf = dft.RKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    Hybrid_Solver(mf, max_cycle_total=30).kernel()

    mf = dft.RKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    mf.kernel()

    # triplet
    mol.spin = 2
    mol.build()
    
    # 2. UKS
    print("\n--- TEST: UKS ---")
    mf = dft.UKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    Hybrid_Solver(mf, max_cycle_total=30).kernel()

    mf = dft.UKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    mf.kernel()

    # 3. ROKS
    print("\n--- TEST: ROKS ---")
    mf = dft.ROKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    Hybrid_Solver(mf, max_cycle_total=30).kernel()

    mf = dft.ROKS(mol).density_fit()
    mf.xc = 'pbe0'
    mf.verbose=4
    mf.kernel()




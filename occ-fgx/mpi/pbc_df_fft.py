#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo, estimate_ke_cutoff, error_for_ke_cutoff
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import fft_ao2mo
from pyscf.pbc.df import aft
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import __config__

KE_SCALING = getattr(__config__, 'pbc_df_aft_ke_cutoff_scaling', 0.75)

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):



        #print('pgggg',lib.current_memory()[0])


        #from pyscf.pbc.df import fft_jk
        import fft_occk
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0): # Gamma-point J/K by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)
        #print('1pgggg',lib.current_memory()[0])



        vj = vk = None
        if kpts.shape == (3,):
            vj, vk = fft_jk.get_jk(self, dm, hermi, kpts, kpts_band,
                                   with_j, with_k, exxdiv)
        else:
            if with_k:
                print('kkkocc',lib.current_memory()[0])
                vk = fft_occk.get_k_kpts_occ(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                print('kkkj',lib.current_memory()[0])
                vj = fft_occk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

def get_jk_orig(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        #from pyscf.pbc.df import fft_jk
        import fft_occk
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0): # Gamma-point J/K by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            vj, vk = fft_jk.get_jk(self, dm, hermi, kpts, kpts_band,
                                   with_j, with_k, exxdiv)
        else:
            if with_k:
                print('kkkorig')
                vk = fft_occk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = fft_occk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk




import dm2c_jk

from pyscf.dft.numint import NumInt
import num_int
NumInt.nr_rks = num_int.NumInt.nr_rks
NumInt.nr_uks = num_int.NumInt.nr_uks
NumInt.__init__ = num_int.NumInt.__init__
NumInt.block_loop_incore = num_int.NumInt.block_loop_incore

import df_grad_rks
from pyscf.df.grad import rks
rks.Gradients.grad_elec = df_grad_rks.grad_elec
rks.Gradients.get_veff = df_grad_rks.get_veff

import df_grad_uks
from pyscf.df.grad import uks
uks.Gradients.grad_elec = df_grad_uks.grad_elec
uks.Gradients.get_veff = df_grad_uks.get_veff

import df_grad_rhf
from pyscf.df.grad.rhf import Gradients
Gradients.grad_elec = df_grad_rhf.grad_elec

import df_grad_uhf
from pyscf.df.grad.uhf import Gradients
Gradients.grad_elec = df_grad_uhf.grad_elec


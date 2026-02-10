import time
from pyscf import gto
import dscf_gpu

mol = gto.M(atom='''
C    0.00000000            0.00000000           -0.60298508
O    0.00000000            0.00000000            0.60539399
H    0.00000000            0.93467313           -1.18217476
H    0.00000000           -0.93467313           -1.18217476
''',
            verbose = 4,
            basis='aug-cc-pvtz')
print('basis=',mol.basis,'nao',mol.nao)

mf = dscf_gpu.DSCF(mol)
#mf.max_cycle = 100
#mf.auxbasis = None

mf.mthd = 'CUKS' #'UKS' #'ROKS' #   'SFRO' #
mf.excite = 'pd' #'mom' # 'vlv' #'chp_ord'  #'cp_ord' #'ex-lshift' #'orig' #'pd_chp' need modify pd_gpu now# 
if mf.excite == 'vlv' or mf.excite == 'ex-lshift': 
  mf.excite_factor = 0.3
mf.xc = 'pbe0'
mf.dips = 'Schmidt' #'Lowdin' # 'Orig' # 

h = homo = mol.nelec[0]-1
l = h + 1
print('homo=',homo)
#mf.d = [[h,l]]
mf.s = [[h,l]]
#mf.s = [[h,l],[h-1,l],[h-2,l],[h,l+1],[h,l+2],[h,l+3],[h,l+4]]
mf.run()












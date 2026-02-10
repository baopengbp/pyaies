#f2py -c --f90flags='-fopenmp' -lgomp -llapack kspies_fort.f90 -m kspies_fort
#f2py --fcompiler=intelem -c jacobi.f90 -m jacobi_dg

f2py -c jacobi.f90 -m jacobi_dg

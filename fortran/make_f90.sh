f2py --fcompiler=intelem --f90flags="-heap-arrays -fopenmp" \
 -liomp5 -lifcoremt -lmkl_core -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_def  -lpthread\
 -I$MKLROOT/interfaces/blas95/include/intel64/lp64 \
 $MKLROOT/interfaces/blas95/lib/intel64/libmkl_blas95_lp64.a \
 -m spec_func -c spectral_function_openmp.f90

f2py --fcompiler=intelem --f90flags="-heap-arrays -fopenmp" \
 -liomp5 -lifcoremt -lmkl_core -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_def  -lpthread\
 -I$MKLROOT/interfaces/blas95/include/intel64/lp64 \
 $MKLROOT/interfaces/blas95/lib/intel64/libmkl_blas95_lp64.a \
 -m get_dos -c get_dos.f90
#
#f2py --fcompiler=intelem --f90flags="-heap-arrays -fopenmp" \
# -liomp5 -lifcoremt -lmkl_core -lmkl_intel_lp64  -lmkl_intel_thread -lmkl_def  -lpthread\
# -I/home/gyu/storage/software/intel/compilers_and_libraries_2018.3.222/linux/mkl/interfaces/blas95/include/intel64/lp64 \
# /home/gyu/storage/software/intel/compilers_and_libraries_2018.3.222/linux/mkl/interfaces/blas95/lib/intel64/libmkl_blas95_lp64.a \
# -m ac_calc -c ac_calc.f90

#pragma once
// includes CUDA
#include <cuda_runtime.h>


// includes, project
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper functions for SDK examples

struct Triangle_GPU{
		unsigned int nVert[3];
		float X[3];
		float Y[3];
};

void PreComputeTriangle(unsigned int*, double*, unsigned int, unsigned int, Triangle_GPU*);

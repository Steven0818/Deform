#include "triangle.h"
#include <helper_functions.h> // helper functions for SDK examples
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <time.h>
#include "iostream"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"


using namespace std;


__global__ void PreComputeTriangle_GPU(unsigned int* m_vVertexMap,double* m_mFirstMatrix,unsigned int row,unsigned int col,Triangle_GPU* m_vTriangles,Vertex_GPU* m_Vertex,unsigned int nVerts){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx>=nVerts)return;
	
	Vertex_GPU & v = m_Vertex[idx];
	
	
	for (int i = 0; i < v.nTriangle; i++){
		Triangle_GPU & t = m_vTriangles[v.Triangles[i]];
		int j = 0;
		//printf("#%d v %d %d %d",idx,);
		for (j; j < 3; j++){
			if (idx == t.nVert[j])
				break;
		}
		if (j == 3){ printf("vertex error idx %d nVert %d %d %d\n", idx, t.nVert[0], t.nVert[1], t.nVert[2]); return; }
		
		int n0x = 2 * m_vVertexMap[t.nVert[j]];
		int n0y = n0x + 1;
		int n1x = 2 * m_vVertexMap[t.nVert[(j + 1) % 3]];
		int n1y = n1x + 1;
		int n2x = 2 * m_vVertexMap[t.nVert[(j + 2) % 3]];
		int n2y = n2x + 1;

		float x0 = t.X[j];
		float y0 = t.Y[j];
		float x1 = t.X[(j + 2) % 3];
		float y1 = t.Y[(j + 2) % 3];
		float x2 = t.X[(j + 1) % 3];
		float y2 = t.Y[(j + 1) % 3];

		//for n0

		m_mFirstMatrix[n0x*row + n0x] += 1 - 2 * x0 + x0*x0 + y0*y0;
		m_mFirstMatrix[n0x*row + n1x] += 2 * x0 - 2 * x0*x0 - 2 * y0*y0;		//m_mFirstMatrix[n1x][n0x] += 2*x - 2*x*x - 2*y*y;
		m_mFirstMatrix[n0x*row + n1y] += 2 * y0;						//m_mFirstMatrix[n1y][n0x] += 2*y;
		m_mFirstMatrix[n0x*row + n2x] += -2 + 2 * x0;					//m_mFirstMatrix[n2x][n0x] += -2 + 2*x;
		m_mFirstMatrix[n0x*row + n2y] += -2 * y0;

		m_mFirstMatrix[n0y*row + n0y] += 1 - 2 * x0 + x0*x0 + y0*y0;
		m_mFirstMatrix[n0y*row + n1x] += -2 * y0;						//m_mFirstMatrix[n1x][n0y] += -2*y;
		m_mFirstMatrix[n0y*row + n1y] += 2 * x0 - 2 * x0*x0 - 2 * y0*y0;		//m_mFirstMatrix[n1y][n0y] += 2*x - 2*x*x - 2*y*y;
		m_mFirstMatrix[n0y*row + n2x] += 2 * y0;						//m_mFirstMatrix[n2x][n0y] += 2*y;
		m_mFirstMatrix[n0y*row + n2y] += -2 + 2 * x0;

		//for n1
		// n1x,n?? elems
		m_mFirstMatrix[n0x*row + n0x] += x1*x1 + y1*y1;
		m_mFirstMatrix[n0x*row + n1x] += -2 * x1;						//m_mFirstMatrix[n2x][n1x] += -2*x;
		m_mFirstMatrix[n0x*row + n1y] += 2 * y1;						//m_mFirstMatrix[n2y][n1x] += 2*y;


		//n1y,n?? elems
		m_mFirstMatrix[n0y*row + n0y] += x1*x1 + y1*y1;
		m_mFirstMatrix[n0y*row + n1x] += -2 * y1;						//m_mFirstMatrix[n2x][n1y] += -2*y;
		m_mFirstMatrix[n0y*row + n1y] += -2 * x1;						//m_mFirstMatrix[n2y][n1y] += -2*x;

		//for n2
		// final 2 elems
		m_mFirstMatrix[n0x*row + n0x] += 1;
		m_mFirstMatrix[n0y*row + n0y] += 1;
		

	}


	/*double fTriSumErr = 0;
		for ( int j = 0; j < 3; ++j ) {
			double fTriErr = 0;

			int n0x = 2 * m_vVertexMap[ t.nVert[j] ];
			int n0y = n0x + 1;
			int n1x = 2 * m_vVertexMap[ t.nVert[(j+1)%3] ];
			int n1y = n1x + 1;
			int n2x = 2 * m_vVertexMap[ t.nVert[(j+2)%3] ];
			int n2y = n2x + 1;
			
			/*n0xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[j]];
			//printf("~~~~~~%d", n0xA[threadIdx.x][j]);
			n0yA[threadIdx.x][j] = n0xA[threadIdx.x][j] + 1;
			n1xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[(j + 1) % 3]];
			n1yA[threadIdx.x][j] = n1xA[threadIdx.x][j] + 1;
			n2xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[(j + 2) % 3]];
			n2yA[threadIdx.x][j] = n2yA[threadIdx.x][j] + 1;
			printf("~");


			float x = t.X[j];
			float y = t.Y[j];
			if ((idx == 0||idx==1)&&j==0)
				printf("test GPU value %d,%d n0x %d n0y %d x %d y%d\n",idx,j, n0x, n0y, x, y);
			


			m_mFirstMatrix[n0x*row+n0x] += 1 - 2*x + x*x + y*y;
			m_mFirstMatrix[n0x*row+n1x] += 2*x - 2*x*x - 2*y*y;		//m_mFirstMatrix[n1x][n0x] += 2*x - 2*x*x - 2*y*y;
			m_mFirstMatrix[n0x*row+n1y] += 2*y;						//m_mFirstMatrix[n1y][n0x] += 2*y;
			m_mFirstMatrix[n0x*row+n2x] += -2 + 2*x;					//m_mFirstMatrix[n2x][n0x] += -2 + 2*x;
			m_mFirstMatrix[n0x*row+n2y] += -2 * y;						//m_mFirstMatrix[n2y][n0x] += -2 * y;



			// n0y,n?? elems
			m_mFirstMatrix[n0y*row+n0y] += 1 - 2*x + x*x + y*y;
			m_mFirstMatrix[n0y*row+n1x] += -2*y;						//m_mFirstMatrix[n1x][n0y] += -2*y;
			m_mFirstMatrix[n0y*row+n1y] += 2*x - 2*x*x - 2*y*y;		//m_mFirstMatrix[n1y][n0y] += 2*x - 2*x*x - 2*y*y;
			m_mFirstMatrix[n0y*row+n2x] += 2*y;						//m_mFirstMatrix[n2x][n0y] += 2*y;
			m_mFirstMatrix[n0y*row+n2y] += -2 + 2*x;					//m_mFirstMatrix[n2y][n0y] += -2 + 2*x;



			// n1x,n?? elems
			m_mFirstMatrix[n1x*row+n1x] += x*x + y*y;
			m_mFirstMatrix[n1x*row+n2x] += -2*x;						//m_mFirstMatrix[n2x][n1x] += -2*x;
			m_mFirstMatrix[n1x*row+n2y] += 2*y;						//m_mFirstMatrix[n2y][n1x] += 2*y;


			//n1y,n?? elems
			m_mFirstMatrix[n1y*row+n1y] += x*x + y*y;
			m_mFirstMatrix[n1y*row+n2x] += -2*y;						//m_mFirstMatrix[n2x][n1y] += -2*y;
			m_mFirstMatrix[n1y*row+n2y] += -2*x;						//m_mFirstMatrix[n2y][n1y] += -2*x;



			// final 2 elems
			m_mFirstMatrix[n2x*row+n2x] += 1;
			m_mFirstMatrix[n2y*row+n2y] += 1;

		}
		*/


		//_RMSInfo("  Total Error: %f\n", fTriSumErr);

		if (idx == 1)
			printf("output test GPU %lf", m_mFirstMatrix[0]);
}

__global__ void nodiag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == i && x != y){
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}

}

__global__ void diag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == y && x == i){
			I[x*n + y] /= A[i*n + i];
			A[x*n + y] /= A[i*n + i];
		}

}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}
		}
	}

}

__global__ void set_zero(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

void PreComputeTriangle(unsigned int*m_vVertexMap_GPU, double* m_mFirstMatrix, unsigned int row, unsigned int col, Triangle_GPU* m_vTriangles,Vertex_GPU* m_Vertex, unsigned int nVerts){
	PreComputeTriangle_GPU<<< nVerts/64+1,64 >>>(m_vVertexMap_GPU, m_mFirstMatrix, row, col, m_vTriangles, m_Vertex, nVerts);
}

void inverse(double* matrix,int n,double* invMat){
	clock_t t1, t2;
	double* mGPrime_host = matrix;
	double* mGPrime_device;
	cudaMalloc((void**)&mGPrime_device, n*n*sizeof(double));
	cudaMemcpy(mGPrime_device, matrix, n*n*sizeof(double), cudaMemcpyHostToDevice);


	
	double* I_device;
	cudaMalloc((void**)&I_device, n*n*sizeof(double));
	double* I = new double[n*n];
	t1 = clock();
	for (int i = 0; i<n; i++){
		for (int j = 0; j<n; j++){
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}
	t2 = clock();
	printf("for loop cost %lf ms\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
	cudaMemcpy(I_device, I, n*n*sizeof(double), cudaMemcpyHostToDevice);
	int blocksize = 16;
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
	for (int i = 0; i<n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(mGPrime_device, I_device, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(mGPrime_device, I_device, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(mGPrime_device, I_device, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(mGPrime_device, I_device, n, i);
	}
	cudaMemcpy(invMat, I_device, n*n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(I_device);
	cudaFree(mGPrime_device);

	//cudaMemcpy(I, d_A, n*n*sizeof(double), cudaMemcpyDeviceToHost);
}
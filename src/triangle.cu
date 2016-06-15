#include "triangle.h"
#include <helper_functions.h> // helper functions for SDK examples
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "iostream"
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;


__global__ void PreComputeTriangle_GPU(unsigned int* m_vVertexMap, unsigned int row, unsigned int col, int nTriangles, Triangle_GPU* m_vTriangles, int* n0xA_device, int *n0yA_device, int* n1yA_device, int*
 n1xA_device,int* n2xA_device,int* n2yA_device,double* n0x_n0xData_device,double* n0x_n1xData_device,double* n0x_n1yData_device,double* n0x_n2xData_device,double* n0x_n2yData_device,double* n0y_n0yData_device,double* n0y_n1xData_device,double* n0y_n1yData_device
,double* n0y_n2xData_device,double* n0y_n2yData_device,double* n1x_n1xData_device,double* n1x_n2xData_device,double* n1x_n2yData_device,double* n1y_n1yData_device,double* n1y_n2xData_device,double* n1y_n2yData_device,double* n2x_n2xData_device
,double* n2y_n2yData_device){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nTriangles)return;
	

	Triangle_GPU & t = m_vTriangles[idx];
	/*extern __shared__  int n0xA[][3];
	extern __shared__  int n0yA[][3];
	extern __shared__  int n1yA[][3];
	extern __shared__  int n1xA[][3];
	extern __shared__  int n2xA[][3];
	extern __shared__  int n2yA[][3];
	extern __shared__  int n0x_n0xData[][3];
	extern __shared__  int n0x_n1xData[][3];
	extern __shared__  int n0x_n1yData[][3];
	extern __shared__  int n0x_n2xData[][3];
	extern __shared__  int n0x_n2yData[][3];
	extern __shared__  int n0y_n0yData[][3];
	extern __shared__  int n0y_n1xData[][3];
	extern __shared__  int n0y_n1yData[][3];
	extern __shared__  int n0y_n2xData[][3];
	extern __shared__  int n0y_n2yData[][3];
	extern __shared__  int n1x_n1xData[][3];
	extern __shared__  int n1x_n2xData[][3];
	extern __shared__  int n1x_n2yData[][3];
	extern __shared__  int n1y_n1yData[][3];
	extern __shared__  int n1y_n2xData[][3];
	extern __shared__  int n1y_n2yData[][3];
	extern __shared__  int n2x_n2xData[][3];
	extern __shared__  int n2y_n2yData[][3];*/

	double fTriSumErr = 0;
	for ( int j = 0; j < 3; ++j ) {
			double fTriErr = 0;

			int n0x = 2 * m_vVertexMap[ t.nVert[j] ];
			int n0y = n0x + 1;
			int n1x = 2 * m_vVertexMap[ t.nVert[(j+1)%3] ];
			int n1y = n1x + 1;
			int n2x = 2 * m_vVertexMap[ t.nVert[(j+2)%3] ];
			int n2y = n2x + 1;
			
			n0xA_device[3*idx+j] = 2 * m_vVertexMap[t.nVert[j]];
			//printf("~~~~~~%d", n0xA[threadIdx.x][j]);
			n0yA_device[3*idx+j] = n0xA_device[3*threadIdx.x+j] + 1;
			n1xA_device[3*idx+j] = 2 * m_vVertexMap[t.nVert[(j + 1) % 3]];
			n1yA_device[3*idx + j] = n1xA_device[3 * threadIdx.x + j] + 1;
			n2xA_device[3*idx + j] = 2 * m_vVertexMap[t.nVert[(j + 2) % 3]];
			n2yA_device[3 *idx + j] = n2xA_device[3 * threadIdx.x + j] + 1;
			//printf("~");


			float x = t.X[j];
			float y = t.Y[j];
			if ((idx == 0||idx==1)&&j==0)
				printf("GPU test test GPU value %d,%d n0x:%d n0y:%d x:%d y:%d\n",idx,j, n0x, n0y, x, y);
			

			n0x_n0xData_device[3 * idx + j] = 1 - 2 * x + x*x + y*y;
			n0x_n1xData_device[3 * idx + j] = 2 * x - 2 * x*x - 2 * y*y;		//m_mFirstMatrix[n1x][n0x] += 2*x - 2*x*x - 2*y*y;
			n0x_n1yData_device[3 * idx + j] = 2 * y;						//m_mFirstMatrix[n1y][n0x] += 2*y;
			n0x_n2xData_device[3 * idx + j] = -2 + 2 * x;					//m_mFirstMatrix[n2x][n0x] += -2 + 2*x;
			n0x_n2yData_device[3 * idx + j] = -2 * y;						//m_mFirstMatrix[n2y][n0x] += -2 * y;



			// n0y,n?? elems
			n0y_n0yData_device[3 * idx + j] = 1 - 2 * x + x*x + y*y;
			n0y_n1xData_device[3 * idx + j] = -2 * y;						//m_mFirstMatrix[n1x][n0y] += -2*y;
			n0y_n1yData_device[3 * idx + j] = 2 * x - 2 * x*x - 2 * y*y;		//m_mFirstMatrix[n1y][n0y] += 2*x - 2*x*x - 2*y*y;
			n0y_n2xData_device[3 * idx + j] = 2 * y;						//m_mFirstMatrix[n2x][n0y] += 2*y;
			n0y_n2yData_device[3 * idx + j] = -2 + 2 * x;					//m_mFirstMatrix[n2y][n0y] += -2 + 2*x;



	// n1x,n?? elems
			n1x_n1xData_device[3 * idx + j] = x*x + y*y;
			n1x_n2xData_device[3 * idx + j] = -2 * x;						//m_mFirstMatrix[n2x][n1x] += -2*x;
			n1x_n2yData_device[3 * idx + j] = 2 * y;						//m_mFirstMatrix[n2y][n1x] += 2*y;


			//n1y,n?? elems
			n1y_n1yData_device[3 *idx+ j] = x*x + y*y;
			n1y_n2xData_device[3 * idx + j] = -2 * y;						//m_mFirstMatrix[n2x][n1y] += -2*y;
			n1y_n2yData_device[3 * idx + j] = -2 * x;						//m_mFirstMatrix[n2y][n1y] += -2*x;



			// final 2 elems
			n2x_n2xData_device[3 * idx + j] = 1;
			n2y_n2yData_device[3 * idx + j] = 1;

			/*m_mFirstMatrix[n0x*row+n0x] += 1 - 2*x + x*x + y*y;
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
			m_mFirstMatrix[n2y*row+n2y] += 1;*/

		}


		//_RMSInfo("  Total Error: %f\n", fTriSumErr);
		/*if (threadIdx.x == 0){
			for (int i = 0; i < row*col; i++){
				for (int j = 0; j < 3; j++){
			
					m_mFirstMatrix[n0xA[i][j] * row + n0xA[i][j]] += n0x_n0xData[i][j];
					m_mFirstMatrix[n0xA[i][j] * row + n1xA[i][j]] += n0x_n1xData[i][j];
					m_mFirstMatrix[n0xA[i][j] * row + n1yA[i][j]] += n0x_n1yData[i][j];
					m_mFirstMatrix[n0xA[i][j] * row + n2xA[i][j]] += n0x_n2xData[i][j];
					m_mFirstMatrix[n0xA[i][j] * row + n2yA[i][j]] += n0x_n2yData[i][j];

					m_mFirstMatrix[n0yA[i][j] * row + n0yA[i][j]] += n0y_n0yData[i][j];
					m_mFirstMatrix[n0yA[i][j] * row + n1xA[i][j]] += n0y_n1xData[i][j];
					m_mFirstMatrix[n0yA[i][j] * row + n1yA[i][j]] += n0y_n1yData[i][j];
					m_mFirstMatrix[n0yA[i][j] * row + n2xA[i][j]] += n0y_n2xData[i][j];
					m_mFirstMatrix[n0yA[i][j] * row + n2yA[i][j]] += n0y_n2yData[i][j];

					m_mFirstMatrix[n1xA[i][j] * row + n1xA[i][j]] += n1x_n1xData[i][j];
					m_mFirstMatrix[n1xA[i][j] * row + n2xA[i][j]] += n1x_n2xData[i][j];
					m_mFirstMatrix[n1xA[i][j] * row + n2yA[i][j]] += n1x_n2yData[i][j];

					m_mFirstMatrix[n1yA[i][j] * row + n1yA[i][j]] += n1y_n1yData[i][j];
					m_mFirstMatrix[n1yA[i][j] * row + n2xA[i][j]] += n1y_n2xData[i][j];
					m_mFirstMatrix[n1yA[i][j] * row + n2yA[i][j]] += n1y_n2yData[i][j];

					m_mFirstMatrix[n2xA[i][j] * row + n2xA[i][j]] += n2x_n2xData[i][j];
					m_mFirstMatrix[n2yA[i][j] * row + n2yA[i][j]] += n2y_n2yData[i][j];
				}

			}
		}*/
		//if (idx == 1)
			//printf("output test GPU %lf", n2x_n2xData_device[3 * threadIdx.x]);
}

void PreComputeTriangle(unsigned int*m_vVertexMap_GPU, double* m_mFirstMatrix, unsigned int row, unsigned int col,int  nTriangles, Triangle_GPU* m_vTriangles){
	unsigned int num = nTriangles;
	int* n0xA = new int[num*3];
	int* n0yA = new int[num * 3];
	int* n1yA = new int[num*3];
	int* n1xA = new int[num*3];
	int* n2xA = new int[num*3];
	int* n2yA = new int[num*3];
	double* n0x_n0xData = new double[num*3];
	double* n0x_n1xData = new double[num*3];
	double* n0x_n1yData = new double[num*3];
	double* n0x_n2xData = new double[num*3];
	double* n0x_n2yData = new double[num*3];
	double* n0y_n0yData = new double[num*3];
	double* n0y_n1xData = new double[num*3];
	double* n0y_n1yData = new double[num*3];
	double* n0y_n2xData = new double[num*3];
	double* n0y_n2yData = new double[num*3];
	double* n1x_n1xData = new double[num*3];
	double* n1x_n2xData = new double[num*3];
	double* n1x_n2yData = new double[num*3];
	double* n1y_n1yData = new double[num*3];
	double* n1y_n2xData = new double[num*3];
	double* n1y_n2yData = new double[num*3];
	double* n2x_n2xData = new double[num*3];
	double* n2y_n2yData = new double[num*3];
	int* n0xA_device;
	int* n0yA_device;
	int* n1yA_device;
	int* n1xA_device;
	int* n2xA_device;
	int* n2yA_device;
	double* n0x_n0xData_device;
	double* n0x_n1xData_device;
	double* n0x_n1yData_device;
	double* n0x_n2xData_device;
	double* n0x_n2yData_device;
	double* n0y_n0yData_device;
	double* n0y_n1xData_device;
	double* n0y_n1yData_device;
	double* n0y_n2xData_device;
	double* n0y_n2yData_device;
	double* n1x_n1xData_device;
	double* n1x_n2xData_device;
	double* n1x_n2yData_device;
	double* n1y_n1yData_device;
	double* n1y_n2xData_device;
	double* n1y_n2yData_device;
	double* n2x_n2xData_device;
	double* n2y_n2yData_device;
	cudaMalloc((void**)&n0xA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n0yA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n1yA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n1xA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n2xA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n2yA_device,sizeof(int)*num*3);
	cudaMalloc((void**)&n0x_n0xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0x_n1xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0x_n1yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0x_n2xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0x_n2yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0y_n0yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0y_n1xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0y_n1yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0y_n2xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n0y_n2yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1x_n1xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1x_n2xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1x_n2yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1y_n1yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1y_n2xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n1y_n2yData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n2x_n2xData_device,sizeof(double)*num*3);
	cudaMalloc((void**)&n2y_n2yData_device,sizeof(double)*num*3);

	PreComputeTriangle_GPU << <nTriangles / 128 + 1, 128 >> >(m_vVertexMap_GPU, row, col, nTriangles, m_vTriangles, n0xA_device, n0yA_device, n1yA_device,
n1xA_device,n2xA_device,n2yA_device,n0x_n0xData_device,n0x_n1xData_device,n0x_n1yData_device,n0x_n2xData_device,n0x_n2yData_device,n0y_n0yData_device,n0y_n1xData_device,n0y_n1yData_device
,n0y_n2xData_device,n0y_n2yData_device,n1x_n1xData_device,n1x_n2xData_device,n1x_n2yData_device,n1y_n1yData_device,n1y_n2xData_device,n1y_n2yData_device,n2x_n2xData_device
,n2y_n2yData_device);

	
	cudaMemcpy(n0xA, n0xA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0yA, n0yA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1xA, n1xA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1yA, n1yA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n2xA, n2xA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n2yA, n2yA_device, sizeof(int)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0x_n0xData, n0x_n0xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0x_n1xData, n0x_n1xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0x_n1yData, n0x_n1yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0x_n2xData, n0x_n2xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0x_n2yData, n0x_n2yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0y_n0yData, n0y_n0yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0y_n1xData, n0y_n1xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n0y_n1yData, n0y_n1yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1x_n1xData, n1x_n1xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1x_n2xData, n1x_n2xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1x_n2yData, n1x_n2yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1y_n1yData, n1y_n1yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1y_n2xData, n1y_n2xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n1y_n2yData, n1y_n2yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n2x_n2xData, n2x_n2xData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(n2y_n2yData, n2y_n2yData_device, sizeof(double)*num * 3, cudaMemcpyDeviceToHost); 
	for (int i = 0; i <nTriangles; i++){
		for (int j = 0; j < 3; j++){
			//printf("testtest %d  %d", n0xA[3 * i + j],n0xA[3 * i + j]);

			m_mFirstMatrix[n0xA[3 * i + j] * row + n0xA[3 * i + j]] += n0x_n0xData[3 * i + j];

			m_mFirstMatrix[n0xA[3 * i + j] * row + n1xA[3 * i + j]] += n0x_n1xData[3 * i + j];

			m_mFirstMatrix[n0xA[3 * i + j] * row + n1yA[3 * i + j]] += n0x_n1yData[3 * i + j];

			m_mFirstMatrix[n0xA[3 * i + j] * row + n2xA[3 * i + j]] += n0x_n2xData[3 * i + j];

			m_mFirstMatrix[n0xA[3 * i + j] * row + n2yA[3 * i + j]] += n0x_n2yData[3 * i + j];
			m_mFirstMatrix[n0yA[3 * i + j] * row + n0yA[3 * i + j]] += n0y_n0yData[3 * i + j];
			m_mFirstMatrix[n0yA[3 * i + j] * row + n1xA[3 * i + j]] += n0y_n1xData[3 * i + j];
			m_mFirstMatrix[n0yA[3 * i + j] * row + n1yA[3 * i + j]] += n0y_n1yData[3 * i + j];
			m_mFirstMatrix[n0yA[3 * i + j] * row + n2xA[3 * i + j]] += n0y_n2xData[3 * i + j];
			m_mFirstMatrix[n0yA[3 * i + j] * row + n2yA[3 * i + j]] += n0y_n2yData[3 * i + j];
			m_mFirstMatrix[n1xA[3 * i + j] * row + n1xA[3 * i + j]] += n1x_n1xData[3 * i + j];
			m_mFirstMatrix[n1xA[3 * i + j] * row + n2xA[3 * i + j]] += n1x_n2xData[3 * i + j];
			m_mFirstMatrix[n1xA[3 * i + j] * row + n2yA[3 * i + j]] += n1x_n2yData[3 * i + j];
			m_mFirstMatrix[n1yA[3 * i + j] * row + n1yA[3 * i + j]] += n1y_n1yData[3 * i + j];
			m_mFirstMatrix[n1yA[3 * i + j] * row + n2xA[3 * i + j]] += n1y_n2xData[3 * i + j];
			m_mFirstMatrix[n1yA[3 * i + j] * row + n2yA[3 * i + j]] += n1y_n2yData[3 * i + j];
			m_mFirstMatrix[n2xA[3 * i + j] * row + n2xA[3 * i + j]] += n2x_n2xData[3 * i + j];
			m_mFirstMatrix[n2yA[3 * i + j] * row + n2yA[3 * i + j]] += n2y_n2yData[3 * i + j];
			if (i == 0 && j == 0){
					printf("test each GPU %f\n", n0x_n0xData[3 * i + j]);
					printf("test each GPU %f\n", n0x_n1xData[3 * i + j]);
					printf("test each GPU %f\n", n0x_n1yData[3 * i + j]);
					printf("test each GPU %f\n", n0x_n2xData[3 * i + j]);
					printf("test each GPU %f\n", n0x_n2yData[3 * i + j]);
				}
			//printf("point GPU value:%f\n", m_mFirstMatrix[0]);
		}

	}
	cudaFree(n0xA_device);
	cudaFree(n0yA_device);
	cudaFree(n1yA_device);
	cudaFree(n1xA_device);
	cudaFree(n2xA_device);
	cudaFree(n2yA_device);
	cudaFree(n0x_n0xData_device);
	cudaFree(n0x_n1xData_device);
	cudaFree(n0x_n1yData_device);
	cudaFree(n0x_n2xData_device);
	cudaFree(n0x_n2yData_device);
	cudaFree(n0y_n0yData_device);
	cudaFree(n0y_n1xData_device);
	cudaFree(n0y_n1yData_device);
	cudaFree(n1x_n1xData_device);
	cudaFree(n1x_n2xData_device);
	cudaFree(n1x_n2yData_device);
	cudaFree(n1y_n1yData_device);
	cudaFree(n1y_n2xData_device);
	cudaFree(n1y_n2yData_device);
	cudaFree(n2x_n2xData_device);
	cudaFree(n2y_n2yData_device);
	delete n0xA;
	delete n0yA;
	delete n1yA;
	delete n1xA;
	delete n2xA;
	delete n2yA;
	delete n0x_n0xData;
	delete n0x_n1xData;
	delete n0x_n1yData;
	delete n0x_n2xData;
	delete n0x_n2yData;
	delete n0y_n0yData;
	delete n0y_n1xData;
	delete n0y_n1yData;
	delete n0y_n2xData;
	delete n0y_n2yData;
	delete n1x_n1xData;
	delete n1x_n2xData;
	delete n1x_n2yData;
	delete n1y_n1yData;
	delete n1y_n2xData;
	delete n1y_n2yData;
	delete n2x_n2xData;
	delete n2y_n2yData;
}
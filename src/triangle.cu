#include "triangle.h"
#include <helper_functions.h> // helper functions for SDK examples
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "iostream"
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;


__global__ void PreComputeTriangle_GPU(unsigned int* m_vVertexMap,double* m_mFirstMatrix,unsigned int row,unsigned int col,Triangle_GPU* m_vTriangles){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx>=row*col)return;
	

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
			
			/*n0xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[j]];
			//printf("~~~~~~%d", n0xA[threadIdx.x][j]);
			n0yA[threadIdx.x][j] = n0xA[threadIdx.x][j] + 1;
			n1xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[(j + 1) % 3]];
			n1yA[threadIdx.x][j] = n1xA[threadIdx.x][j] + 1;
			n2xA[threadIdx.x][j] = 2 * m_vVertexMap[t.nVert[(j + 2) % 3]];
			n2yA[threadIdx.x][j] = n2yA[threadIdx.x][j] + 1;*/
			printf("~");


			float x = t.X[j];
			float y = t.Y[j];
			if ((idx == 0||idx==1)&&j==0)
				printf("test GPU value %d,%d n0x %d n0y %d x %d y%d\n",idx,j, n0x, n0y, x, y);
			

			/*n0x_n0xData[threadIdx.x][j] = 1 - 2 * x + x*x + y*y;
			n0x_n1xData[threadIdx.x][j] = 2 * x - 2 * x*x - 2 * y*y;		//m_mFirstMatrix[n1x][n0x] += 2*x - 2*x*x - 2*y*y;
			n0x_n1yData[threadIdx.x][j] = 2 * y;						//m_mFirstMatrix[n1y][n0x] += 2*y;
			n0x_n2xData[threadIdx.x][j] = -2 + 2 * x;					//m_mFirstMatrix[n2x][n0x] += -2 + 2*x;
			n0x_n2yData[threadIdx.x][j] = -2 * y;						//m_mFirstMatrix[n2y][n0x] += -2 * y;



			// n0y,n?? elems
			n0y_n0yData[threadIdx.x][j] = 1 - 2 * x + x*x + y*y;
			n0y_n1xData[threadIdx.x][j] = -2 * y;						//m_mFirstMatrix[n1x][n0y] += -2*y;
			n0y_n1yData[threadIdx.x][j] = 2 * x - 2 * x*x - 2 * y*y;		//m_mFirstMatrix[n1y][n0y] += 2*x - 2*x*x - 2*y*y;
			n0y_n2xData[threadIdx.x][j] = 2 * y;						//m_mFirstMatrix[n2x][n0y] += 2*y;
			n0y_n2yData[threadIdx.x][j] = -2 + 2 * x;					//m_mFirstMatrix[n2y][n0y] += -2 + 2*x;



			// n1x,n?? elems
			n1x_n1xData[threadIdx.x][j] = x*x + y*y;
			n1x_n2xData[threadIdx.x][j] = -2 * x;						//m_mFirstMatrix[n2x][n1x] += -2*x;
			n1x_n2yData[threadIdx.x][j] = 2 * y;						//m_mFirstMatrix[n2y][n1x] += 2*y;


			//n1y,n?? elems
			n1y_n1yData[threadIdx.x][j] = x*x + y*y;
			n1y_n2xData[threadIdx.x][j] = -2 * y;						//m_mFirstMatrix[n2x][n1y] += -2*y;
			n1y_n2yData[threadIdx.x][j] = -2 * x;						//m_mFirstMatrix[n2y][n1y] += -2*x;



			// final 2 elems
			n2x_n2xData[threadIdx.x][j] = 1;
			n2y_n2yData[threadIdx.x][j] = 1;*/


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

		__syncthreads();

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
		if (idx == 1)
			printf("output test GPU %lf", m_mFirstMatrix[0]);
}

void PreComputeTriangle(unsigned int*m_vVertexMap_GPU, double* m_mFirstMatrix, unsigned int row, unsigned int col, Triangle_GPU* m_vTriangles){
	unsigned int num = (col*row) / 64 + 1;
	PreComputeTriangle_GPU << <num, 64, col*row *sizeof(int)>> >(m_vVertexMap_GPU, m_mFirstMatrix, row, col, m_vTriangles);
}
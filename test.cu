#include <cuda_runtime.h>
// includes, project
//#include "stdafx.h"
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include "device_launch_parameters.h"
#include "test.h"
#include "iostream"

using namespace std;


class Demo {
	public:
	Demo();
	int* a;
};
Demo::Demo() {
	a = new int[10];
}

__global__ void test2(int* demo){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	printf("fuckyou2");
}

/*__global__ void setDeformHandle(unsigned int* m_vSelected, float* tm_vertices, int offset , rmsmesh::Constraint* m_vConstraints_device, rmsmesh::Vertex* m_vDeformedVerts_device, rmsmesh::RigidMeshDeformer2D* m_deformer, rmsmesh::Constraint* updateConstraints_device, unsigned int m_vConstraintSize){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int vVertex = m_vSelected[idx];
	float x = tm_vertices[vVertex*offset];
	float y = tm_vertices[vVertex*offset+1];
	int count = 0;
	printf("fuck");*/
	/*for (int i = 0; i < m_vConstraintSize; i++){
		if (m_vConstraints_device[i].nVertex == vVertex){
			m_vConstraints_device[i].vConstrainedPos.m_afTuple[0] = x;
			m_vConstraints_device[i].vConstrainedPos.m_afTuple[1] = y;
			m_vDeformedVerts_device[vVertex].vPosition.m_afTuple[0] = x;
			m_vDeformedVerts_device[vVertex].vPosition.m_afTuple[1] = y;
			break;
		}
		count++;
	}
	if (count == m_vConstraintSize){
		updateConstraints_device[idx].nVertex = vVertex;
		updateConstraints_device[idx].vConstrainedPos.m_afTuple[0] = x;
		updateConstraints_device[idx].vConstrainedPos.m_afTuple[1] = y;
		m_vConstraints_device[m_vConstraintSize] = updateConstraints_device[idx];
		m_deformer->m_bSetupValid = false;
		m_vDeformedVerts_device[vVertex].vPosition.m_afTuple[0] = x;
		m_vDeformedVerts_device[vVertex].vPosition.m_afTuple[1] = y;
	}*/
	//(*m_deformedMesh).GetVertex(m_vSelected[idx], vVertex[idx]);
	//m_deformer.SetDeformedHandle(nVertex, Wml::Vector2f(vVertex.X(), vVertex.Y()));
//}


void kernal(){
	

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s



	//Demo* demoTest_host = new Demo;
	int* demoTest_device;
	/*for (int i = 0; i < 10; i++){
		demoTest_host->a[i] = i;
	}*/
	
	
	cudaMalloc(&demoTest_device, 100*sizeof(int));
	test2 << <1, 64 >> >(demoTest_device);


}

void getVertex(std::set<unsigned int> m_vSelected,rmsmesh::TriangleMesh& m_deformedMesh, rmsmesh::RigidMeshDeformer2D& m_deformer){
	
	size_t nConstraints = m_vSelected.size();
	float* tm_vertices_host = &m_deformedMesh.GetVertices()[0];
	float* tm_vertices_device;
	
	cudaMalloc((void**)&tm_vertices_device, 4*nConstraints*sizeof(float));
	cudaMemcpy(tm_vertices_device, tm_vertices_host , 4 * nConstraints*sizeof(float), cudaMemcpyHostToDevice);

	
	
	std::vector<unsigned int> tmp_m_vSelected(nConstraints);
	std::copy(m_vSelected.begin(), m_vSelected.end(), tmp_m_vSelected.begin());
	unsigned int* m_vSelected_host = new unsigned int[nConstraints];
	std::copy(tmp_m_vSelected.begin(), tmp_m_vSelected.end(), m_vSelected_host);
	unsigned int* m_vSelected_device;
	cudaMalloc((void**)&m_vSelected_device, nConstraints*sizeof(unsigned int));
	cudaMemcpy(m_vSelected_device, m_vSelected_host, nConstraints*sizeof(unsigned int), cudaMemcpyHostToDevice);

	/*
	rmsmesh::TriangleMesh* m_deformedMesh_host = &m_deformedMesh;
	rmsmesh::TriangleMesh* m_deformedMesh_device;

	rmsmesh::RigidMeshDeformer2D* m_deformer_host = &m_deformer;
	rmsmesh::RigidMeshDeformer2D* m_deformer_device;
	


	cudaMalloc((void**)&m_deformedMesh_device, sizeof(rmsmesh::TriangleMesh));
	cudaMemcpy(m_deformedMesh_device, m_deformedMesh_host, sizeof(rmsmesh::TriangleMesh), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_deformer_device, sizeof(rmsmesh::RigidMeshDeformer2D));
	cudaMemcpy(m_deformer_device, m_deformer_host, sizeof(rmsmesh::RigidMeshDeformer2D), cudaMemcpyHostToDevice);
	*/
	/*std::set<rmsmesh::Constraint> m_vconstraints = m_deformer.getm_vConstraints();
	std::vector<rmsmesh::Constraint> tmpm_vconstraints(m_vconstraints.size());
	rmsmesh::Constraint* m_vConstraints_host = new rmsmesh::Constraint[nConstraints+1];
	std::copy(m_vconstraints.begin(), m_vconstraints.end(), tmpm_vconstraints.begin());
	std::copy(tmpm_vconstraints.begin(), tmpm_vconstraints.end(), m_vConstraints_host);
	rmsmesh::Constraint* m_vConstraints_device;
	cudaMalloc((void**)&m_vConstraints_device, (nConstraints+1)*sizeof(rmsmesh::Constraint));
	cudaMemcpy(m_vConstraints_device, m_vConstraints_host, (nConstraints + 1)*sizeof(rmsmesh::Constraint), cudaMemcpyHostToDevice);

	std::vector<rmsmesh::Vertex> m_vDeformedVerts = m_deformer.getm_vDeformedVerts();
	rmsmesh::Vertex* m_vDeformedVerts_host = new rmsmesh::Vertex[m_vDeformedVerts.size()];
	std::copy(m_vDeformedVerts.begin(), m_vDeformedVerts.end(), m_vDeformedVerts_host);
	rmsmesh::Vertex* m_vDeformedVerts_device;
	cudaMalloc((void**)&m_vDeformedVerts_device, m_vDeformedVerts.size()*sizeof(rmsmesh::Vertex));
	cudaMemcpy(m_vDeformedVerts_device, m_vDeformedVerts_host, m_vDeformedVerts.size()*sizeof(rmsmesh::Vertex), cudaMemcpyHostToDevice);

	rmsmesh::RigidMeshDeformer2D* m_derformer_device;
	cudaMalloc((void**)&m_derformer_device, sizeof(rmsmesh::RigidMeshDeformer2D));
	cudaMemcpy(m_derformer_device, &m_deformer, sizeof(rmsmesh::RigidMeshDeformer2D), cudaMemcpyHostToDevice);

	rmsmesh::Constraint* updateConstraints_device;
	cudaMalloc((void**)&updateConstraints_device, nConstraints*sizeof(rmsmesh::Constraint));
	printf("12312");
	setDeformHandle << <nConstraints / 128 + 1, 128 >> >(m_vSelected_device, tm_vertices_device, 4 , m_vConstraints_device, m_vDeformedVerts_device, m_derformer_device, updateConstraints_device, nConstraints);*/
//	std::set<unsigned int>::iterator cur(m_vSelected.begin()), end(m_vSelected.end());
	// if a lot of points , need add something in  
	 
	/*while (cur != end) {
		m_deformer.SetDeformedHandle(nVertex, Wml::Vector2f(vVertex.X(), vVertex.Y()));
	} */
	
}

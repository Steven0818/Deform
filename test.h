#include "RigidMeshDeformer2D.h"
extern "C"  void kernal();
extern "C"  void getVertex(std::set<unsigned int> m_vSelected, rmsmesh::TriangleMesh& m_deformedMesh, rmsmesh::RigidMeshDeformer2D& m_deformer);
//void hello();
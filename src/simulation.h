#ifndef SIMULATION_H
#define SIMULATION_H

typedef unsigned int TColor;

#define BLOCKDIM_X 20
#define BLOCKDIM_Y 20


#define DELTA 0.016

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions

extern "C" void cuda_calculate(float* array_v,float* array_c, int num_of_vectors,int num_of_clusters);
extern "C" void cuda_calculate_clusters(float* array_v,float* array_c, int num_of_vectors,int num_of_clusters);
#endif

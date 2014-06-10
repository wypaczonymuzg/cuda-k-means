// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
GLuint shader;
#define PI 3.1415926535897932384626433832795
#define LENGTH 5000

int sh_size_v;
int sh_size_c;
int num_of_vectors;
int num_of_clusters;
float* h_array_v;
float* h_array_c;
float* d_array_v;
float* d_array_c;
float delta;
//int zal = 0;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int g_Kernel = 0;

unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

void displayCUDAInfo() {
	const int kb = 1024;
	const int mb = kb * kb;
	printf("CUDA INFO:\n=========\n\nCUDA version:   v%d\n", CUDART_VERSION);

	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Devices: \n\n");

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("%d : %s:%d.%d\n", i, props.name, props.major, props.minor);
		printf("  Global memory:   %dmb\n", props.totalGlobalMem / mb);
		printf("  Shared memory:   %dkb\n", props.sharedMemPerBlock / kb);
		printf("  Constant memory: %dkb\n", props.totalConstMem / kb);
		printf("  Block registers: %d\n", props.regsPerBlock);

		printf("  Warp size:         %d\n", props.warpSize);
		printf("  Threads per block: %d\n", props.maxThreadsPerBlock);
		printf("  Max block dimensions: [ %d, %d, %d ]\n",
				props.maxThreadsDim[0], props.maxThreadsDim[1],
				props.maxThreadsDim[2]);
		printf("  Max grid dimensions:  [ %d, %d, %d ]\n\n=========\n\n",
				props.maxGridSize[0], props.maxGridSize[1],
				props.maxGridSize[2]);
	}
}

void runSimulation(float threshold) {
	float changesFac = (float) num_of_vectors;
	float *hn_array_v;
	float* hn_array_c;
	hn_array_v = (float*) malloc(sh_size_v);
	hn_array_c = (float*) malloc(sh_size_c);

	switch (g_Kernel) {
	case 0:
		break;
	case 1:
		while (changesFac > threshold) {

			cuda_calculate(d_array_v, d_array_c, num_of_vectors,
					num_of_clusters);
			cuda_calculate_clusters(d_array_v, d_array_c, num_of_vectors,
					num_of_clusters);

			checkCudaErrors(
					cudaMemcpy(hn_array_v, d_array_v,
							num_of_vectors * 4 * sizeof(float),
							cudaMemcpyDeviceToHost));

			checkCudaErrors(
					cudaMemcpy(hn_array_c, d_array_c, sh_size_c,
							cudaMemcpyDeviceToHost));
			int counter = 0;
			for (int i = 0; i < num_of_clusters; i++) {
				int idx = i * 4;
				if (hn_array_c[idx + 3] != h_array_c[idx + 3])
					counter++;
				printf("i = %d \thn_array_c[idx+3] = %f\n", i,
						hn_array_c[idx + 3]);
			}
			checkCudaErrors(
					cudaMemcpy(h_array_c, d_array_c, sh_size_c,
							cudaMemcpyDeviceToHost));
			changesFac = (float) (counter * 1.0f / num_of_clusters * 1.0f);
			printf(
					"counter = %d\t numof clusters  = %d\tchanges factor  = %f\n",
					counter, num_of_clusters, changesFac);
		}
		/*
		 printf("___________________________________after kernel usage\n");
		 for (int i = 0; i < num_of_vectors * 4; i += 4) {
		 printf("i : %d\t x : %f\t y : %f\t z : %f\t cluster  : %f\n", i / 4,
		 hn_array_v[i], hn_array_v[i + 1], hn_array_v[i + 2],
		 hn_array_v[i + 3]);
		 }
		 printf("*************************clusters***********************\n");

		 for (int i = 0; i < num_of_clusters * 4; i += 4) {
		 printf("host i : %d\t x : %f\t y : %f\t z : %f\t4  : %f\n", i / 4,
		 hn_array_c[i], hn_array_c[i + 1], hn_array_c[i + 2],
		 hn_array_c[i + 3]);
		 }
		 */
		break;
	}

	getLastCudaError("Filtering kernel execution failed.\n");
}

float RandomFloat(float a, float b) {
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

int main(int argc, char **argv) {

//float* array;
	pArgc = &argc;
	pArgv = argv;
	float threshold = 1;
	displayCUDAInfo();
	g_Kernel = 0;
	printf("starting...\n\n");

	printf("choose simulation : \n1 : draw 1 \n2 : none\n3 : none\n");
	int n = 1;
	//scanf("%d", &n);
	switch (n) {
	case 1:
		threshold = 0.0f;
		num_of_vectors = 3000;
		num_of_clusters = 1024;

		sh_size_v = sizeof(float) * num_of_vectors * 4;
		h_array_v = (float*) malloc(sh_size_v);
		sh_size_c = sizeof(float) * num_of_clusters * 4;
		h_array_c = (float*) malloc(sh_size_c);

		printf("number of vectors = %d \t sh_size_v = %d \n", num_of_vectors,
				sh_size_v);

		for (int i = 0; i < num_of_vectors * 4; i += 4) {
			h_array_v[i] = RandomFloat(0, LENGTH); 		//x
			h_array_v[i + 1] = RandomFloat(0, LENGTH); 	//y
			h_array_v[i + 2] = RandomFloat(0, LENGTH);	//z
			h_array_v[i + 3] = 0;
			printf("i : %d\t x : %f\t y : %f\t z : %f\tcluster  = %f\n", i / 4,
					h_array_v[i], h_array_v[i + 1], h_array_v[i + 2],
					h_array_v[i + 3]);
		}

		printf("\n\n\nCLUSTERS!!!!!!\n\n\n");

		printf("number of clusters = %d \t sh_size_c = %d \n", num_of_clusters,
				sh_size_c);

		for (int i = 0; i < num_of_clusters * 4; i += 4) {
			if (i < num_of_vectors) {
				h_array_c[i] = h_array_v[i] + 1; 		//x
				h_array_c[i + 1] = h_array_v[i + 1]; 	//y
				h_array_c[i + 2] = h_array_v[i + 2];	//z
			} else {
				h_array_c[i] = RandomFloat(LENGTH, LENGTH); 		//x
				h_array_c[i + 1] = RandomFloat(LENGTH, LENGTH); 	//y
				h_array_c[i + 2] = RandomFloat(LENGTH, LENGTH);	//z

			}
			h_array_c[i + 3] = 0;
			printf("i : %d\t x : %f\t y : %f\t z : %f\t clusters : %f\n", i / 4,
					h_array_c[i], h_array_c[i + 1], h_array_c[i + 2],
					h_array_c[i + 3]);
		}

		g_Kernel = 1;
		break;
	default:
		g_Kernel = 0;
		break;
	}

//cudaGLSetGLDevice (gpuGetMaxGflopsDeviceId());
	/*
	 h_Src = (uchar4*) malloc(imageH * imageW * 4);
	 memset(h_Src, clearColorbit, imageH * imageW * 4);


	 checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));
	 */
//copying to device memory
	checkCudaErrors(cudaMalloc((void** ) &d_array_v, sh_size_v));
	checkCudaErrors(
			cudaMemcpy(d_array_v, h_array_v, sh_size_v,
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void** ) &d_array_c, sh_size_c));
	checkCudaErrors(
			cudaMemcpy(d_array_c, h_array_c, sh_size_c,
					cudaMemcpyHostToDevice));
	runSimulation(threshold);

// cudaDeviceReset causes the driver to clean up all state. While
// not mandatory in normal operation, it is good practice.  It is also
// needed to ensure correct operation when the application is being
// profiled. Calling cudaDeviceReset causes all profile data to be
// flushed before the application exits
	cudaDeviceReset();
	return 0;
	exit(EXIT_SUCCESS);
}


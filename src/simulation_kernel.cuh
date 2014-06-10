#define RADIUS 5
#define ALIGN 4
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line,
		bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

extern __shared__ float sh_memory[];

__global__ void calculate(float* array_v, float* array_c, int num_of_vectors,
		int num_of_clusters) {
int threads = BLOCKDIM_X*BLOCKDIM_Y;
	int id = (blockIdx.x *threads + threadIdx.y * BLOCKDIM_Y + threadIdx.x);
	int gtid = 4 * id;
	float myPosition[4];
	//float myPositionC;
	//printf("gtid = %d \t array_v[gtid] =%f\tarray_v[gtid+1] =%f\tarray_v[gtid+2] =%f\tarray_v[gtid+3] =%d\n",gtid,array_v[gtid],array_v[gtid+1],array_v[gtid+2],array_v[gtid+3]);
	myPosition[0] = array_v[gtid];
	myPosition[1] = array_v[gtid + 1];
	myPosition[2] = array_v[gtid + 2];
	myPosition[3] = array_v[gtid + 3];

	for (int i = 0; i < num_of_clusters; i++) {
		int idx = i * 4;
		sh_memory[idx] = array_c[idx];
		sh_memory[idx + 1] = array_c[idx + 1];
		sh_memory[idx + 2] = array_c[idx + 2];
		sh_memory[idx + 3] = array_c[idx + 3];

	}
	__syncthreads();
	int cluster = (int)myPosition[3] * 4;

	float xd = myPosition[0] - sh_memory[cluster];
	float yd = myPosition[1] - sh_memory[cluster + 1];
	float zd = myPosition[2] - sh_memory[cluster + 2];

	float myDistSq = xd * xd + yd * yd + zd * zd;

	for (int i = 0; i < num_of_clusters; i++) {
		int idx = i * 4;
		xd = myPosition[0] - sh_memory[idx];
		yd = myPosition[1] - sh_memory[idx + 1];
		zd = myPosition[2] - sh_memory[idx + 2];

		float distSq = xd * xd + yd * yd + zd * zd;

		if (distSq < myDistSq) {
			myDistSq= distSq;
			myPosition[3] = (float)i;
		}

	}

	__syncthreads();
	array_v[gtid + 3] = myPosition[3];

}
__global__ void calculate_clusters(float* array_v, float* array_c,
		int num_of_vectors, int num_of_clusters) {
	int threads = BLOCKDIM_X*BLOCKDIM_Y;
		int id = (blockIdx.x *threads + threadIdx.y * blockDim.y + threadIdx.x);
		int gtid = 4 * id;

	//int gtid = 4*id;
	float myPosition[4];

	myPosition[0] = array_c[gtid];
	myPosition[1] = array_c[gtid + 1];
	myPosition[2] = array_c[gtid + 2];
	myPosition[3] = array_c[gtid + 3];


	for (int i = 0; i < num_of_vectors; i++) {
		int idx = i * 4;
		sh_memory[idx] = array_v[idx];
		sh_memory[idx + 1] = array_v[idx + 1];
		sh_memory[idx + 2] = array_v[idx + 2];
		sh_memory[idx + 3] = array_v[idx + 3];
	}

	__syncthreads();

	float xd = 0;
	float yd = 0;
	float zd = 0;

	int number = 0;

	for (int i = 0; i < num_of_vectors; i++) {
		int idx = i * 4;


		if ( sh_memory[idx + 3] == (float)id*1.0f) {
			xd += sh_memory[idx];
			yd += sh_memory[idx + 1];
			zd += sh_memory[idx + 2];
			number++;
		}
	}
	__syncthreads();
	printf("id = %d \t number = %d\n",id,number);
	if (number*1.0f != 0.0f) {
		xd = xd / number;
		yd = yd / number;
		zd = zd / number;


		array_c[gtid] = xd;
		array_c[gtid + 1] = yd;
		array_c[gtid + 2] = zd;
		array_c[gtid +3] =number;

	}

}

extern "C" void cuda_calculate(float* array_v, float *array_c,
		int num_of_vectors, int num_of_clusters) {

	int blocks = num_of_vectors / (BLOCKDIM_X * BLOCKDIM_Y);

	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

	dim3 grid(blocks);

	unsigned int aligned = num_of_clusters;
	aligned += ALIGN - aligned % ALIGN;

	int sh_size = aligned * sizeof(float) * 4;

	calculate<<<grid, threads, sh_size>>>(array_v, array_c, num_of_vectors,
			num_of_clusters);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

}
extern "C" void cuda_calculate_clusters(float* array_v, float *array_c,
		int num_of_vectors, int num_of_clusters) {

	int blocks = num_of_clusters / (BLOCKDIM_X * BLOCKDIM_Y);
	if (blocks == 0)
		blocks++;


	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);

	dim3 grid(blocks);

	unsigned int aligned = num_of_vectors;
	aligned += ALIGN - aligned % ALIGN;

	int sh_size = aligned * sizeof(float) * 4;



	calculate_clusters<<<grid, threads, sh_size>>>(array_v, array_c,
			num_of_vectors, num_of_clusters);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

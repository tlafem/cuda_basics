#include "l2_norm.cuh"

#define BLOCKSIZE 1024

__global__ void l2_norm_dVector_kernel(double *a, double *partial_sum, int n) {
	__shared__ double partial_sums[BLOCKSIZE];

	double local_sum = 0;

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int partial_index = threadIdx.x;

	while (id < n) {
		local_sum += (a[id] * a[id]);
		id += (blockDim.x * gridDim.x); // this thread may have to handle multiple sums
	}

	partial_sums[partial_index] = local_sum;

	__syncthreads();

	int sum_level = blockDim.x >> 1; // divide by 2

	while (sum_level != 0) {
		if (partial_index < sum_level) {
			partial_sums[partial_index] += partial_sums[partial_index + sum_level];
		}

		__syncthreads();

		sum_level >>= 1; // divide by 2
	}

	if (partial_index == 0) {
		// if we are the thread processing index 0 of partial_sums for our block
		partial_sum[blockIdx.x] = partial_sums[0];
	}
	// at this point there is still some partial somes left to compute
	// inefficient to do so on GPU. Let CPU do this
}

int l2_norm_dVector(dVector a, double *result) {
	double *device_a, *host_partial, *device_partial;
	int sizeInBytes = a.len * sizeof(double);
	int err = 0;

	cudaError_t cudaStatus;

	int gridSize = (int)ceil((float)a.len / BLOCKSIZE);

	host_partial = (double *)calloc(gridSize, sizeof(double));

	cudaStatus = cudaMalloc(&device_a, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "A: cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc(&device_partial, gridSize * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "device_partial: cudaMalloc failed!\n");
		goto Error;
	}

	cudaMemcpy(device_a, a.data, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "A: cudaMemcpy to device failed!\n");
		goto Error;
	}

	l2_norm_dVector_kernel<<<gridSize, BLOCKSIZE>>>(device_a, device_partial, a.len);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "magnitude_dVector_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaMemcpy(host_partial, device_partial, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "partial: cudaMemcpy to host failed!\n");
		goto Error;
	}

Error:
	cudaFree(device_a);
	cudaFree(device_partial);
	if (cudaStatus != cudaSuccess) {
		err = -1;
		goto Exit;
	}

	double sum = 0;
	for (int i = 0; i < gridSize; ++i) {
		sum += host_partial[i];
	}

	*result = sqrt((double)sum);

Exit:
	free(host_partial);
	return err;
}

void test_l2_norm(int n) {
	dVector a;
	dvector_init(a, n);

	double L2_norm = 0;
	double euclidean_norm = 0;

	for (int i = 0; i < n; ++i) {
		a.data[i] = 1;
	}

	l2_norm_dVector(a, &L2_norm);
	euclidean_norm_dVector(a, &euclidean_norm);

	//L2 norm should be sqrt(n)
	if (abs(L2_norm - sqrt((double)n)) > 1e-14) {
		fprintf(stdout, "L2_norm=%f is not correct within error.\n", L2_norm);
	}
	else {
		fprintf(stdout, "L2_norm=%f is correct within error.\n", L2_norm);
	}

	if (abs(L2_norm - euclidean_norm) > 1e-14) {
		fprintf(stdout, "Alias Error: L2_norm not the same as euclidean_norm.\n");
	}
	else {
		fprintf(stdout, "Alias for euclidean_norm to l2_norm is correct.\n");
	}
	
	dvector_free(a);
}

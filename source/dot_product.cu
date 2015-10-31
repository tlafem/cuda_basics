#include "dot_product.cuh"

#define BLOCKSIZE 1024

__global__ void dotProduct_dVector_kernel(double *a, double *b, double *partial_sum, int n) {
	__shared__ double partial_sums[BLOCKSIZE];

	double local_sum = 0;

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int partial_index = threadIdx.x;

	while (id < n) {
		local_sum += (a[id] * b[id]);
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

int dotProduct_dVectors(dVector a, dVector b, double *result) {
	if (a.len != b.len) {
		fprintf(stderr, "Vector length mismatch\n");
		return -1;
	}

	int err = 0;
	double *device_a, *device_b, *device_partial, *host_partial;

	int sizeInBytes = a.len * sizeof(double);
	cudaError_t cudaStatus;

	int gridSize = (int)ceil((float)a.len / BLOCKSIZE);

	host_partial = (double *)calloc(gridSize, sizeof(double));

	cudaStatus = cudaMalloc(&device_a, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "A: cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc(&device_b, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "B: cudaMalloc failed!\n");
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

	cudaMemcpy(device_b, b.data, sizeInBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "B: cudaMemcpy to device failed!\n");
		goto Error;
	}

	dotProduct_dVector_kernel<<<gridSize, BLOCKSIZE >>>(device_a, device_b, device_partial, a.len);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "add_dVector_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaFree(device_b);
	cudaFree(device_partial);
	if (cudaStatus != cudaSuccess) {
		err = -1;
		goto Exit;
	}

	double sum = 0;
	for (int i = 0; i < gridSize; ++i) {
		sum += host_partial[i];
	}

	*result = sum;

Exit:
	free(host_partial);
	return err;
}

void test_dotProduct(int n) {
	dVector a, b, c;
	dvector_init(a, n);
	dvector_init(b, n);

	double dotProduct = 0;

	for (int i = 0; i < n; ++i) {
		a.data[i] = 1;
		b.data[i] = 2;
	}

	dotProduct_dVectors(a, b, &dotProduct);

	//dotProduct should be 2*n
	if (abs(dotProduct - 2 * n) > 1e-14) {
		fprintf(stdout, "Dot Product=%f is not correct within error.\n", dotProduct);
	}
	else {
		fprintf(stdout, "Dot Product=%f is correct within error.\n", dotProduct);
	}

	dvector_free(a);
	dvector_free(b);

}
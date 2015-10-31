#include "add.cuh"

#define BLOCKSIZE 1024

__global__ void add_dVector_kernel(double *a, double *b, double *c, int n) {
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < n)
		c[id] = a[id] + b[id];
}

int add_dVectors(dVector a, dVector b, dVector c) {
	if (a.len != b.len || a.len != c.len) {
		fprintf(stderr, "Vector length mismatch\n");
		return -1;
	}

	double *device_a, *device_b, *device_c;
	int sizeInBytes = a.len * sizeof(double);
	cudaError_t cudaStatus;


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

	cudaStatus = cudaMalloc(&device_c, sizeInBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "C: cudaMalloc failed!\n");
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


	int gridSize = (int)ceil((float)a.len / BLOCKSIZE);
	fprintf(stdout, "gridSize=%d, blockSize=%d\n", gridSize, BLOCKSIZE);
	add_dVector_kernel<<<gridSize, BLOCKSIZE>>>(device_a, device_b, device_c, a.len);
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

	cudaMemcpy(c.data, device_c, sizeInBytes, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "C: cudaMemcpy to host failed!\n");
		goto Error;
	}

Error:
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	return (cudaStatus == cudaSuccess)? 0 : -1;
}

void test_add(int n) {
	dVector a, b, c;
	dvector_init(a, n);
	dvector_init(b, n);
	dvector_init(c, n);

	for (int i = 0; i < n; ++i) {
		a.data[i] = .75;
		b.data[i] = .25;
	}

	add_dVectors(a, b, c);

	// every element should be 1
	int errors = 0;
	for (int i = 0; i < n; ++i) {
//		fprintf(stdout, "c[%d]=%f\n", i, c.data[i]);
		if (abs(1 - c.data[i]) > 1e-14) {
			++errors;
		}
	}
	if (errors > 0) {
		fprintf(stdout, "Errors in %d of %d elements\n", errors, n);
	}
	else {
		fprintf(stdout, "No errors in %d elements\n", n);
	}

	dvector_free(a);
	dvector_free(b);
	dvector_free(c);

}
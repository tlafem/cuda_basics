#include "l2_norm.cuh"

int l2_norm_dVectors(dVector a, dVector b, double *result) {
	int error = dotProduct_dVectors(a, b, result);
	if (error != 0)
		return error;
	*result = sqrt(*result);
	return 0;
}

void test_l2_norm(int n) {
	dVector a, b, c;
	dvector_init(a, n);
	dvector_init(b, n);

	double L2_norm = 0;
	double euclidean_norm = 0;

	for (int i = 0; i < n; ++i) {
		a.data[i] = 1;
		b.data[i] = 2;
	}

	l2_norm_dVectors(a, b, &L2_norm);
	euclidean_norm_dVectors(a, b, &euclidean_norm);

	//dotProduct should be 2*n
	if (abs(L2_norm - sqrt((double)(2 * n))) > 1e-14) {
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
	dvector_free(b);
}

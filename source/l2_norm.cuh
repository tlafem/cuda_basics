#ifndef L2_NORM
#define L2_NORM

#include <stdio.h>
#include <math.h>

#include "vector.cuh"
#include "dot_product.cuh"

/**
Alias for l2_norm_dVectors. As such, its parameters are as follows:
dVector a, dVector b, and double *result
*/
#define euclidean_norm_dVectors(a, b, result) l2_norm_dVectors(a, b, result);

int l2_norm_dVectors(dVector a, dVector b, double *result);

void test_l2_norm(int n);

#endif
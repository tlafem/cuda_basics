#ifndef L2_NORM
#define L2_NORM

#include <stdio.h>
#include <math.h>

#include "vector.cuh"

/**
Alias for l2_norm_dVector. As such, its parameters are as follows:
dVector a, and double *result
*/
#define euclidean_norm_dVector(a, result) l2_norm_dVector(a, result);

/**
Computes the standard L2 norm on vector a. Norm is is SUM(a_i * a_i).

*result is a pointer to where the result will be stored
Returns: 0 on success, non-zero otherwise
*/
int l2_norm_dVector(dVector a, double *result);

void test_l2_norm(int n);

#endif
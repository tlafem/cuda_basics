#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include "vector.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

/**
Computes the dot product of two vectors a and b, storing the result in *result.
Dot product is defined as the sum of component pairs, i.e. SUM(a_i * b_i).
Computation happens in stages with only the last stage happening on the CPU, not the GPU.

Vectors must be the same same length.

Returns: 0 on success, non-zero otherwise
*/
int dotProduct_dVectors(dVector a, dVector b, double *result);

void test_dotProduct(int n);

#endif
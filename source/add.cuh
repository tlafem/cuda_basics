#ifndef ADD_H
#define ADD_H

#include "vector.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

/**
Componentwise addition of two vectors a and b, storing the result in a vector c.
Vectors must all be of the same length.

Returns: 0 on success, non-zero otherwise
*/
int add_dVectors(dVector a, dVector b, dVector c);

void test_add(int n);

#endif
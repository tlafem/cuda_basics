#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include "vector.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

int dotProduct_dVectors(dVector a, dVector b, double *result);

void test_dotProduct(int n);

#endif
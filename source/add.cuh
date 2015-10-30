#ifndef ADD_H
#define ADD_H

#include "vector.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

int add_dVectors(dVector a, dVector b, dVector c);

void test_add(int n);

#endif
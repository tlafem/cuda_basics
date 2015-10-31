#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>

/**
Number of components in the vector. If length is 0, vector has not been initialized with any data space
*/
#define dvector_len(v) (v->len)

/**
The index-th entry of the vector if element exists, NULL otherwise
*/
#define dvector_get(v, index) ((index >= v->len)? NULL : (v->data)[index])

/**
Allocates memory for size number of doubles and assigns it to the data field of the vector. All elements initialized to 0.
If not enough memory available, len is set to 0 and data remains unchanged.
*/
#define dvector_init(v, size) v.len = (v.data = (double *)calloc(size, sizeof(double)))? size:0;

/**
Frees the memory associated with the data of the vector, not memory associated with the struct itself. Sets len to 0.
*/
#define dvector_free(v) v.len=0; free(v.data);

typedef struct d_vector_struct {
	double *data;
	int len = 0;
} dVector;


#endif
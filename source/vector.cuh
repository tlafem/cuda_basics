#ifndef VECTOR_H
#define VECTOR_H

#include <stdlib.h>

#define dvector_len(v) (v->len)

#define dvector_get(v, index) ((index >= v->len)? NULL : (v->data)[index])

#define dvector_init(v, size) v.len=size; v.data = (double *)calloc(size, sizeof(double));

#define dvector_free(v) free(v.data);

typedef struct d_vector_struct {
	double *data;
	int len = 0;
} dVector;


#endif
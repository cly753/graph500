#ifndef MPI_BOTTOM_UP_H
#define MPI_BOTTOM_UP_H

#include <stdlib.h>

#include "constants.h"

void one_step_bottom_up();

void one_step_bottom_up_gpu();

void init_bottom_up_gpu();

void end_bottom_up_gpu();

// no need to use if cuda ompi
void pred_to_gpu();

// use this if cuda ompi
void init_pred_gpu(int root);

void pred_from_gpu();

#endif //MPI_BOTTOM_UP_H

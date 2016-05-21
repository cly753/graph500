#ifndef MPI_BOTTOM_UP_H
#define MPI_BOTTOM_UP_H

#include <stdlib.h>

int64_t *frontier;
int64_t *frontier_next;

void init_bottom_up();

int have_more_bottom_up();

void sync_bottom_up();

void one_step_bottom_up();

void switch_to_bottom_up();

#endif //MPI_BOTTOM_UP_H

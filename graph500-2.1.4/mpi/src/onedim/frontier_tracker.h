#ifndef MPI_FRONTIER_TRACKER_H
#define MPI_FRONTIER_TRACKER_H

#include <stdlib.h>

int64_t *frontier;
int64_t *frontier_next;

void init_frontier();

void sync_frontier();

int frontier_have_more();

extern int64_t *frontier_g; // point to memory in GPU
extern int64_t *frontier_next_g; // point to memory in GPU

void sync_frontier_gpu();

void sync_frontier_work_around();

#endif //MPI_FRONTIER_TRACKER_H

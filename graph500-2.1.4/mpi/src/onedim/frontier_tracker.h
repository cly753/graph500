#ifndef MPI_FRONTIER_TRACKER_H
#define MPI_FRONTIER_TRACKER_H

#include <stdlib.h>

int64_t *frontier;
int64_t *frontier_next;


extern int64_t *frontier_g; // pointer to memory in GPU
extern int64_t *frontier_next_g; // pointer to memory in GPU

void init_frontier();

void sync_frontier();

int frontier_have_more();



void set_frontier_gpu(int v);

void sync_frontier_gpu();

int frontier_have_more_gpu();

#endif //MPI_FRONTIER_TRACKER_H

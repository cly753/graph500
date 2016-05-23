#ifndef MPI_FRONTIER_TRACKER_H
#define MPI_FRONTIER_TRACKER_H

#include <stdlib.h>

int64_t *frontier;
int64_t *frontier_next;

void init_frontier();

void sync_frontier();

int frontier_have_more();

#endif //MPI_FRONTIER_TRACKER_H

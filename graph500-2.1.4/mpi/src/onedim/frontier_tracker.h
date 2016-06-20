#ifndef MPI_FRONTIER_TRACKER_H
#define MPI_FRONTIER_TRACKER_H

#include <stdlib.h>

int64_t *frontier;
int64_t *frontier_next;

void init_frontier();

void sync_frontier();

int frontier_have_more();

void add_frontier_next(int64_t to);

#endif //MPI_FRONTIER_TRACKER_H

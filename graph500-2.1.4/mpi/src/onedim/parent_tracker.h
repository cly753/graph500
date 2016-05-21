#ifndef MPI_PARENT_TRACKER_H
#define MPI_PARENT_TRACKER_H

#include <stdlib.h>
#include "constants.h"
#include "print.h"

#include "common.h"

#define size_parent_each (1 << (18)) // worst case: nlocalvert
int64_t size_parent_total;
int64_t size_counter;
int *send_count; // store the number of parent sending out
int *sdispls;
int *receive_count;
int *rdispls;
int64_t *parent_cur; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]
int64_t *parent_next; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]

void show_counter();

void show_displs();

void show_parent();

int have_more();

void add_parent(int64_t from, int64_t to);

void sync();

void init_parent_tracker();

#endif //MPI_PARENT_TRACKER_H

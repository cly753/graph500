#ifndef MPI_BOTTOM_UP_H
#define MPI_BOTTOM_UP_H

#include <stdlib.h>

#include "constants.h"


#define MOD_SIZE_G(v) ((v) & ((1 << lgsize_g) - 1))
#define DIV_SIZE_G(v) ((v) >> lgsize_g)
#define MUL_SIZE_G(x) ((x) << lgsize_g)

#define VERTEX_OWNER_G(v) ((int)(MOD_SIZE_G(v)))
#define VERTEX_LOCAL_G(v) ((size_t)(DIV_SIZE_G(v)))
#define VERTEX_TO_GLOBAL_G(r, i) ((int64_t)(MUL_SIZE_G((uint64_t)i) + (int)(r)))

// #define LONG_BITS_G (sizeof(unsigned long) * CHAR_BIT)
#define LONG_BITS_G 64

#define SET_LOCAL_G(v, a) do {(a)[VERTEX_LOCAL_G((v)) / LONG_BITS_G] |= (1UL << (VERTEX_LOCAL_G((v)) % LONG_BITS_G));} while (0)
#define TEST_LOCA_G(v, a) (((a)[VERTEX_LOCAL_G((v)) / LONG_BITS_G] & (1UL << (VERTEX_LOCAL_G((v)) % LONG_BITS_G))) != 0)

#define SET_GLOBAL_G(v, a) do {(a)[(v) / LONG_BITS_G] |= (1UL << ((v) % LONG_BITS_G));} while (0)
#define SET_GLOBAL_ATOMIC_G(v, a) do {atomicOr((unsigned long long int*)(&(a)[(v) / LONG_BITS_G]), 1UL << ((v) % LONG_BITS_G));} while (0)
#define TEST_GLOBAL_G(v, a) (((a)[(v) / LONG_BITS_G] & (1UL << ((v) % LONG_BITS_G))) != 0)


void one_step_bottom_up();

// gpu related functions 

void one_step_bottom_up_gpu();

// allocate memory for graph, frontier, frontier_next, pred
// copy graph to memory
void init_bottom_up_gpu();

// free memeory
void end_bottom_up_gpu();

void pred_to_gpu();

void init_pred_gpu(int64_t root, int is_root_owner);

void pred_from_gpu();

// frontier_g

void set_frontier_gpu(int64_t v);

int frontier_have_more_gpu();

// workaround for no cuda aware support
void read_frontier_next_g();
void save_frontier_g();

// debug functions
// cpu pointer
int64_t* get_frontier_g();
int64_t* get_frontier_next_g();


#endif //MPI_BOTTOM_UP_H
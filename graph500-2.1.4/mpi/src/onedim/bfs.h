
#ifndef BFS_H
#define BFS_H

#include "limits.h"
#include "common.h"
#include "oned_csr.h"
#include "constants.h"

#define LONG_BITS (64)
#define LONG_BITS_LG (6)
#define LONG_BITS_BIN 0x0000000000003f

#if 1

#define SET_LOCAL(v, a) do {(a)[VERTEX_LOCAL((v)) >> LONG_BITS_LG] |= (1UL << (VERTEX_LOCAL((v)) & LONG_BITS_BIN));} while (0)
#define TEST_LOCAL(v, a) (((a)[VERTEX_LOCAL((v)) >> LONG_BITS_LG] & (1UL << (VERTEX_LOCAL((v)) & LONG_BITS_BIN))) != 0)

#define SET_GLOBAL(v, a) do {(a)[(v) >> LONG_BITS_LG] |= (1UL << ((v) & LONG_BITS_BIN));} while (0)
#define TEST_GLOBAL(v, a) (((a)[(v) >> LONG_BITS_LG] & (1UL << ((v) & LONG_BITS_BIN))) != 0)

#else

#define SET_LOCAL(v, a) do {(a)[VERTEX_LOCAL((v)) / LONG_BITS] |= (1UL << (VERTEX_LOCAL((v)) % LONG_BITS));} while (0)
#define TEST_LOCAL(v, a) (((a)[VERTEX_LOCAL((v)) / LONG_BITS] & (1UL << (VERTEX_LOCAL((v)) % LONG_BITS))) != 0)

#define SET_GLOBAL(v, a) do {(a)[(v) / LONG_BITS] |= (1UL << ((v) % LONG_BITS));} while (0)
#define TEST_GLOBAL(v, a) (((a)[(v) / LONG_BITS] & (1UL << ((v) % LONG_BITS))) != 0)

#endif

#ifdef SHOWTIMER
#define TIME_IT(x, double_ptr_time_output) { \
    if (rank == root_owner) { \
        double time_start = MPI_Wtime(); \
        x; \
        double time_stop = MPI_Wtime(); \
        double diff = time_stop - time_start; \
        PRINTLN_RANK(#x " used %.6lfs", diff); \
        *double_ptr_time_output = diff; \
    } \
    else { \
        x; \
    } \
}
#else
#define TIME_IT(x, double_ptr_time_output) { x; *double_ptr_time_output = 0; }
#endif

#define mpi_assert(expression) \
    do { \
        if (!(expression)) { \
            fprintf(stderr, "Failed assertion at %d in %s",__LINE__, __FILE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)



int local_long_n;
size_t local_long_nb;
int global_long_n;
size_t global_long_nb;

//unsigned long *g_cur;
//unsigned long *g_next;

void show_local(int64_t *a);

void show_global(int64_t *a);

void show_pred();

void bfs(oned_csr_graph* g, int64_t root, int64_t* pred);

#endif // BFS_H
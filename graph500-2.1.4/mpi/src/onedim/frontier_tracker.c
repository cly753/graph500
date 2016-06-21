#include "frontier_tracker.h"

#include <string.h>

#include "oned_csr.h"

#include "bfs.h"
#include "print.h"

void init_frontier() {
    if (frontier == NULL)
        frontier = xmalloc(global_long_nb);
    if (frontier_next == NULL)
        frontier_next = xmalloc(global_long_nb);

    memset(frontier, 0, global_long_nb);
    memset(frontier_next, 0, global_long_nb);
}

void sync_frontier() {
#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: frontier_next:", rank)
    show_global(frontier_next);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // allreduce to broadcast frontier
    MPI_Allreduce(
            frontier_next, // void* send_data
            frontier, // void* recv_data
            global_long_n, // int count
            MPI_LONG, // MPI_Datatype datatype
            MPI_BOR, // MPI_Op op
            MPI_COMM_WORLD // MPI_Comm communicator
    );
    memset(frontier_next, 0, global_long_nb);
#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: frontier:", rank)
    show_global(frontier);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

int frontier_have_more() {
#ifdef USE_OPENMP
    omp_set_num_threads(12);
#endif

    int yes = 0;
    int i;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < global_long_n; i++) {
        if (frontier[i])
#ifdef USE_OPENMP
            yes = 1;
#else
            return 1;
#endif  
    }
    return yes;
}


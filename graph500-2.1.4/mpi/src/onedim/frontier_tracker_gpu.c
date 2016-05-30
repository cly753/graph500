
#include "frontier_tracker.h"

#include "bfs.h"

extern int64_t *frontier_g;
extern int64_t *frontier_next_g;

void set_frontier_gpu(int v) {
	// TODO
	// use cudaMemset
}

void sync_frontier_gpu() {
    // allreduce to broadcast frontier_g
    MPI_Allreduce(
            frontier_next_g, // void* send_data
            frontier_g, // void* recv_data
            global_long_n, // int count
            MPI_LONG, // MPI_Datatype datatype
            MPI_BOR, // MPI_Op op
            MPI_COMM_WORLD // MPI_Comm communicator
    );
    memset(frontier_next_g, 0, global_long_nb);
}

int frontier_have_more_gpu() {
	// TODO
	// use kernel function
}
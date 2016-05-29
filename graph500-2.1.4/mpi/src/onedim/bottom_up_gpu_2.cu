
// 
// CUDA-aware Open MPI
// Now, the Open MPI library will automatically detect that the pointer being passed in is a CUDA device memory pointer and do the right thing. This is referred to as CUDA-aware support.
// https://www.open-mpi.org/faq/?category=runcuda
// https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/
// 

extern "C" {
#include "bottom_up.h"
}

#include <string.h>

#include "oned_csr.h"

#include "constants.h"
#include "bfs.h"
#include "print.h"

// includes CUDA
#include <cuda_runtime.h>

// includes, project
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper functions for SDK examples

// 
// !!! Assume SIZE_MUST_BE_A_POWER_OF_TWO !!!
// 

__device__ int rank_g;
__device__ int size_g;
__device__ int lgsize_g;
__device__ int64_t nlocalverts_g;

struct Context {
	int rank;
	int size;
	int lgsize;
	int64_t nlocalverts;
};

__device__ void set_context(Context context) {
	rank_g = context.rank;
	size_g = context.size;
	lgsize_g = context.lgsize;
	nlocalverts_g = context.nlocalverts;
}

// __device__ int64_t mode_size(int64_t v) {
// 	return v & ((1 << lgsize_g) - 1);
// }

// __device__ int64_t div_size(int64_t v) {
// 	return v >> lgsize_g;
// }

// __device__ int64_t mul_size(int64_t v) {
// 	return v << lgsize_g;
// }

// __device__ int vertex_owner(int64_t v) {
// 	return mode_size(v);
// }

// __device__ size_t vertex_local(int64_t v) {
// 	return div_size(v);
// }

// __device__ int64_t vertex_to_global(int r, int64_t i) {
// 	return mul_size(i) + r;
// }

#define LONG_BITS_G 64

// __device__ void set_local(int64_t v, int64_t *a) { // x / 64 --> x >> 6 ??
// 	a[vertex_local(v) / LONG_BITS_G] |= (1UL << (vertex_local(v) % LONG_BITS_G));
// }

// __device__ int test_local(int64_t v, int64_t *a) {
// 	return 0 != (a[vertex_local(v) / LONG_BITS_G] & (1UL << (vertex_local(v) % LONG_BITS_G)));
// }

// __device__ void set_global(int64_t v, int64_t *a) { // x / 64 --> x >> 6 ??
// 	a[v / LONG_BITS_G] |= (1UL << (v % LONG_BITS_G));
// }

// __device__ void set_global_atomic(int64_t v, int64_t *a) {
// 	atomicOr((unsigned long long int*)(&a[v / LONG_BITS_G]), 1UL << (v % LONG_BITS_G));
// }

// __device__ int test_global(int64_t v, int64_t *a) {
// 	return 0 != (a[v / LONG_BITS_G] & (1UL << (v % LONG_BITS_G)));
// }

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


extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;

#define BLOCK_X 64

// do_nothing kernel
// for bottom up
// one thread do one vertex
// one thread try to find parent in its all neighbour
void dim_do_nothing(dim3* dimGrid, dim3 *dimBlock) {
	*dimGrid = dim3((g.nlocalverts + BLOCK_X - 1) / BLOCK_X, 1, 1); // number of block
	*dimBlock = dim3(BLOCK_X, 1, 1); // number of thread per block
}

__global__ void do_nothing(
	int64_t *rowstarts_g, 
	int64_t *column_g, 
	int64_t *frontier_g, 
	int64_t *frontier_next_g, 
	int64_t *pred_g, 
	Context context) {

	set_context(context);

	const int block_base = blockIdx.x * blockDim.x;
	const int64_t i = block_base + threadIdx.x;

    __syncthreads();

	if (i >= nlocalverts_g)
		return ;

    if (pred_g[i] == -1) {
        int j;
        for (j = (int) rowstarts_g[i]; j < rowstarts_g[i + 1]; j++) {
            int64_t parent_global = column_g[j];        
            if (TEST_GLOBAL_G(parent_global, frontier_g)) {
                pred_g[i] = parent_global;
                SET_GLOBAL_ATOMIC_G(VERTEX_TO_GLOBAL_G(rank_g, i), frontier_next_g);
                break;
            }
        }
    }
}

int64_t *rowstarts_g;
int size_rowstarts;
int64_t *column_g;
int size_column;

int64_t *pred_g;
int size_pred_g;

int64_t *frontier_g;
int size_frontier_g;
int64_t *frontier_next_g;
int size_frontier_next_g;


// transfer graph to gpu global memory
// should perform only once
void init_bottom_up_gpu() {
	size_rowstarts = (g.nlocalverts + 1) * sizeof(int64_t);
	size_column = g.rowstarts[g.nlocalverts] * sizeof(int64_t);
	cudaMalloc((void **)&rowstarts_g, size_rowstarts);
	cudaMalloc((void **)&column_g, size_column);
	cudaMemcpy(rowstarts_g, g.rowstarts, size_rowstarts, cudaMemcpyHostToDevice);
	cudaMemcpy(column_g, g.column, size_column, cudaMemcpyHostToDevice);

	// here assume pred always reside in GPU
	// from beginning to end
	// only when everythiing is done
	// transfer pred back to CPU
	size_pred_g = g.nlocalverts * sizeof(int64_t);
	cudaMalloc((void **)&pred_g, size_pred_g);
	// cudaMemcpy(pred_g, pred, size_pred_g, cudaMemcpyHostToDevice);

	size_frontier_g = global_long_nb;
	cudaMalloc((void **)&frontier_g, size_frontier_g);
	size_frontier_next_g = global_long_nb;
	cudaMalloc((void **)&frontier_next_g, size_frontier_next_g);
}

void pred_to_gpu() {
	cudaMemcpy(pred_g, pred, size_pred_g, cudaMemcpyHostToDevice);
}

void pred_from_gpu() {
	cudaMemcpy(pred, pred_g, size_pred_g, cudaMemcpyDeviceToHost);
}

void end_bottom_up_gpu() {
	cudaFree(rowstarts_g);
	cudaFree(column_g);
	cudaFree(pred_g);
	cudaFree(frontier_g);
	cudaFree(frontier_next_g);
}

void one_step_bottom_up_gpu() {
	// transfer current frontier to gpu
	cudaMemcpy(frontier_g, frontier, size_frontier_g, cudaMemcpyHostToDevice);
	cudaMemset(frontier_next_g, 0, size_frontier_next_g);

	// get suitable dim
	dim3 dimGrid;
	dim3 dimBlock;
	dim_do_nothing(&dimGrid, &dimBlock);

	// launch gpu kernel
	// it should compute frontier_next_g
	Context context = { rank, size, lgsize, g.nlocalverts };
	do_nothing<<<dimGrid, dimBlock>>>(rowstarts_g, column_g, frontier_g, frontier_next_g, pred_g, context);

	cudaMemcpy(frontier_next, frontier_next_g, size_frontier_next_g, cudaMemcpyDeviceToHost);
}

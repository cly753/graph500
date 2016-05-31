
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
#include "frontier_tracker.h"
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

// __device__ int64_t vertex_to_global(int rank, int64_t v) {
// 	return mul_size((uint64_t)v) + rank;
// }

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


extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;

#define BLOCK_X 64

// do_nothing kernel
// for bottom up
// one thread do one vertex
// one thread try to find parent in its all neighbour

// get dim for bfs kernel
void dim_do_nothing(dim3* dimGrid, dim3 *dimBlock) {
	*dimGrid = dim3((g.rowstarts[g.nlocalverts] + BLOCK_X - 1) / BLOCK_X, 1, 1); // number of block
	*dimBlock = dim3(BLOCK_X, 1, 1); // number of thread per block
}

// bfs kernel
// each thread do one edge
__global__ void do_nothing(
	int64_t *row_g, 
	int64_t *column_g, 
	int64_t *frontier_g, 
	int64_t *frontier_next_g, 
	int64_t *pred_g,
	int64_t total_edge,
	Context context) {

	set_context(context);

	const int block_base = blockIdx.x * blockDim.x;
	const int i = block_base + threadIdx.x; // this thread do i-th edge

	if (i >= total_edge)
		return ;

	int from = row_g[i]; // one end of the edge
	if (pred_g[from] == -1) { // bottom up, so check if from is unvisited
		int to_global = column_g[i]; // the other end of the edge
		if (TEST_GLOBAL_G(to_global, frontier_g)) { // check if is in frontier
			pred_g[from] = to_global;
			SET_GLOBAL_ATOMIC_G(VERTEX_TO_GLOBAL_G(rank_g, from), frontier_next_g);
		}
	}
}

int64_t *rowstarts_g;
int size_rowstarts;
int64_t *column_g;
int size_column;

int64_t *row_g;
int size_row;

int64_t *pred_g;
int size_pred_g;

int64_t *frontier_g;
int size_frontier_g;
int64_t *frontier_next_g;
int size_frontier_next_g;

void show_pred_g() {
	int64_t *pred_copy = (int64_t *)xmalloc(size_pred_g);
	cudaMemcpy(pred_copy, pred_g, size_pred_g, cudaMemcpyDeviceToHost);

    PRINT_RANK("gpu index:")
    for (int i = 0; i < g.nlocalverts; i++) {
        PRINT(" %"PRId64"", (i * size + rank))
    }
    PRINTLN("")
    PRINT_RANK("gpu pred :")
    for (int i = 0; i < g.nlocalverts; i++) {
        PRINT(" %"PRId64"", pred_copy[i])
    }
    PRINTLN("")

    free(pred_copy);
}

void read_frontier_next_g() {
	cudaMemcpy(frontier_next, frontier_next_g, global_long_nb, cudaMemcpyDeviceToHost);
}

void save_frontier_g() {
	cudaMemcpy(frontier_g, frontier, global_long_nb, cudaMemcpyHostToDevice);
}

int64_t* get_frontier_g() {
	int64_t *frontier_g_copy = (int64_t *)xmalloc(global_long_nb);
	cudaMemcpy(frontier_g_copy, frontier_g, global_long_nb, cudaMemcpyDeviceToHost);	
	return frontier_g_copy;
}

int64_t* get_frontier_next_g() {
	int64_t *frontier_next_g_copy = (int64_t *)xmalloc(global_long_nb);
	cudaMemcpy(frontier_next_g_copy, frontier_next_g, global_long_nb, cudaMemcpyDeviceToHost);
	return frontier_next_g_copy;
}

// use CPU to fill?
__global__ void fill_row_g(int64_t *rowstarts_g, int64_t *row_g, int nlocalverts) {
	const int block_base = blockIdx.x * blockDim.x;
	const int i = block_base + threadIdx.x;
	if (i >= nlocalverts)
		return ;
	for (int j = rowstarts_g[i]; j < rowstarts_g[i + 1]; j++)
		row_g[j] = i;
}

__global__ void fill_row_g_binary(int64_t *rowstarts_g, int64_t *row_g, int nlocalverts, int64_t total_edge) {
	const int block_base = blockIdx.x * blockDim.x;
	const int i = block_base + threadIdx.x;
	if (i >= total_edge)
		return ;
	int l = 0;
	int r = nlocalverts;
	// int r = i;
	while (1) {
		int m = (l + r) / 2;
		int a = rowstarts_g[m];
		int b = rowstarts_g[m + 1];
		if (a <= i && i < b) {
			row_g[i] = m;
			break;
		}
		else if (b <= i) {
			l = m + 1;
		}
		else {
			r = m;
		}
	}
}

// transfer graph to gpu global memory
// should perform only once
void init_bottom_up_gpu() {
	size_rowstarts = (g.nlocalverts + 1) * sizeof(int64_t);
	size_column = g.rowstarts[g.nlocalverts] * sizeof(int64_t);
	cudaMalloc((void **)&rowstarts_g, size_rowstarts);
	cudaMalloc((void **)&column_g, size_column);
	cudaMemcpy(rowstarts_g, g.rowstarts, size_rowstarts, cudaMemcpyHostToDevice);
	cudaMemcpy(column_g, g.column, size_column, cudaMemcpyHostToDevice);

	size_row = size_column;
	cudaMalloc((void **)&row_g, size_row);
	fill_row_g<<<(g.nlocalverts + BLOCK_X - 1) / BLOCK_X, BLOCK_X>>>(rowstarts_g, row_g, g.nlocalverts);
	// fill_row_g_binary<<<(g.rowstarts[g.nlocalverts] + BLOCK_X - 1) / BLOCK_X, BLOCK_X>>>(rowstarts_g, row_g, g.nlocalverts, g.rowstarts[g.nlocalverts]);

	// here assume pred always reside in GPU
	// from beginning to end
	// only when everythiing is done
	// transfer pred back to CPU
	size_pred_g = g.nlocalverts * sizeof(int64_t);
	cudaMalloc((void **)&pred_g, size_pred_g);
	// cudaMemcpy(pred_g, pred, size_pred_g, cudaMemcpyHostToDevice);

	size_frontier_g = global_long_nb;
	cudaMalloc((void **)&frontier_g, size_frontier_g);
	cudaMemset(frontier_g, 0, size_frontier_g);
	size_frontier_next_g = global_long_nb;
	cudaMalloc((void **)&frontier_next_g, size_frontier_next_g);
}

// no need to use if cuda ompi
void pred_to_gpu() {
	cudaMemcpy(pred_g, pred, size_pred_g, cudaMemcpyHostToDevice);
}

// use this if cuda ompi
void init_pred_gpu(int64_t root, int is_root_owner) {
	cudaMemset(pred_g, -1, size_pred_g);
	if (is_root_owner) {
		// http://stackoverflow.com/questions/7464015/cuda-change-single-value-in-array
		cudaMemcpy(pred_g + VERTEX_LOCAL(root), &root, sizeof(int64_t), cudaMemcpyHostToDevice); 
	}
#ifdef SHOWDEBUG
	show_pred_g();
#endif
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

	cudaFree(row_g);
}

void one_step_bottom_up_gpu() {
	cudaMemset(frontier_next_g, 0, size_frontier_next_g);

	// get suitable dim
	dim3 dimGrid;
	dim3 dimBlock;
	dim_do_nothing(&dimGrid, &dimBlock);

	// launch gpu kernel
	// it should compute frontier_next_g
	Context context = { rank, size, lgsize, g.nlocalverts };
	do_nothing<<<dimGrid, dimBlock>>>(row_g, column_g, frontier_g, frontier_next_g, pred_g, g.rowstarts[g.nlocalverts], context);
#ifdef SHOWDEBUG
	show_pred_g();
#endif
}

void set_frontier_gpu(int64_t v) {
	// http://stackoverflow.com/questions/7464015/cuda-change-single-value-in-array
	int the_long = v / LONG_BITS_G;
	int64_t val = 1UL << (v % LONG_BITS_G);
	cudaMemcpy(frontier_g + the_long, &val, sizeof(int64_t), cudaMemcpyHostToDevice); 
}

__device__ int have_more_g;

__global__ void check_have_more(int64_t *frontier_g, int global_long_n) {
	const int block_base = blockIdx.x * blockDim.x;
	const int i = block_base + threadIdx.x;
	if (i >= global_long_n)
		return ;

	__shared__ int hm;
	hm = 0;
	__syncthreads();
	if (frontier_g[i])
		hm = 1;
	__syncthreads();
	if (threadIdx.x == 0 && hm)
		have_more_g = 1;
}

// __global__ void reset_have_more_g() {
// 	have_more_g = 0;
// }

#define HAVE_MORE_BLOCK_SIZE 64

// gpu version of checking if frontier is empty
// so check if all frontier int64_t is 0 or not
int frontier_have_more_gpu() {
	int have_more = 0;

	int grid_size = (global_long_n + HAVE_MORE_BLOCK_SIZE - 1) / HAVE_MORE_BLOCK_SIZE;
	// reset_have_more_g<<<1, 1>>>();
	cudaMemcpyToSymbol(have_more_g, &have_more, sizeof(int), 0, cudaMemcpyHostToDevice);
	check_have_more<<<grid_size, HAVE_MORE_BLOCK_SIZE>>>(frontier_g, global_long_n);

	cudaMemcpyFromSymbol(&have_more, have_more_g, sizeof(int), 0, cudaMemcpyDeviceToHost);
	return have_more;
}

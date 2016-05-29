
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


#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
#define MOD_SIZE_G(v) ((v) & ((1 << lgsize_g) - 1))
#define DIV_SIZE_G(v) ((v) >> lgsize_g)
#define MUL_SIZE_G(x) ((x) << lgsize_g)
#else
#define MOD_SIZE_G(v) ((v) % size_g)
#define DIV_SIZE_G(v) ((v) / size_g)
#define MUL_SIZE_G(x) ((x) * size_g)
#endif
#define VERTEX_OWNER_G(v) ((int)(MOD_SIZE_G(v)))
#define VERTEX_LOCAL_G(v) ((size_t)(DIV_SIZE_G(v)))
#define VERTEX_TO_GLOBAL_G(r, i) ((int64_t)(MUL_SIZE_G((uint64_t)i) + (int)(r)))

// #define LONG_BITS_G (sizeof(unsigned long) * CHAR_BIT)
#define LONG_BITS_G 64

#define SET_LOCAL_G(v, a) do {(a)[VERTEX_LOCAL_G((v)) / LONG_BITS_G] |= (1UL << (VERTEX_LOCAL_G((v)) % LONG_BITS_G));} while (0)
#define TEST_LOCA_G(v, a) (((a)[VERTEX_LOCAL_G((v)) / LONG_BITS_G] & (1UL << (VERTEX_LOCAL_G((v)) % LONG_BITS_G))) != 0)

#define SET_GLOBAL_G(v, a) do {(a)[(v) / LONG_BITS_G] |= (1UL << ((v) % LONG_BITS_G));} while (0)
#define TEST_GLOBAL_G(v, a) (((a)[(v) / LONG_BITS_G] & (1UL << ((v) % LONG_BITS_G))) != 0)

// #define SET_G(v, a) do {(a) |= (1UL << (v));} while (0)
// #define TEST_G(v, a) (((a) & (1UL << (v))) != 0)


extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;


__device__ int rank_g;
__device__ int size_g;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
__device__ int lgsize_g;
#endif
__device__ int64_t nlocalverts_g;


struct Context {
	int rank;
	int size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	int lgsize;
#endif
	int64_t nlocalverts;
};

__device__ void set_context(Context context) {
	rank_g = context.rank;
	size_g = context.size;
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
	lgsize_g = context.lgsize;
#endif
	nlocalverts_g = context.nlocalverts;
}



#define BLOCK_X 64

// do_nothing kernel
// for bottom up
// one thread do one vertex
// one thread try to find parent in its all neighbour
void dim_do_nothing(dim3* dimGrid, dim3 *dimBlock) {
	*dimGrid = dim3(128, 1, 1); // number of block
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
	__shared__ char next[BLOCK_X];
	next[threadIdx.x] = 0;

    __syncthreads();

	if (i >= nlocalverts_g)
		return ;

    if (pred_g[i] == -1) {
        int j;
        for (j = (int) rowstarts_g[i]; j < rowstarts_g[i + 1]; j++) {
            int64_t parent_global = column_g[j];        
            if (TEST_GLOBAL_G(parent_global, frontier_g)) {
                pred_g[i] = parent_global;
                next[threadIdx.x] = 1;
                break;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
    	int x;
    	for (x = 0; x < BLOCK_X; x++) {
    		if (next[x] == 1) {
	    		int real_i = block_base + x;
	            SET_GLOBAL_G(VERTEX_TO_GLOBAL_G(rank_g, real_i), frontier_next_g);	
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
	cudaMemcpy(pred_g, pred, size_pred_g, cudaMemcpyHostToDevice);

	size_frontier_g = global_long_nb;
	cudaMalloc((void **)&frontier_g, size_frontier_g);
	size_frontier_next_g = global_long_nb;
	cudaMalloc((void **)&frontier_next_g, size_frontier_next_g);
}

void end_bottom_up_gpu() {
	cudaMemcpy(pred, pred_g, size_pred_g, cudaMemcpyDeviceToHost);

	cudaFree(rowstarts_g);
	cudaFree(column_g);
	cudaFree(pred_g);
	cudaFree(frontier_g);
	cudaFree(frontier_next_g);
}

// entry to do one level bfs on gpu
// it should transfer graph to gpu global memory
// possibly only once
// and compute suitable grid size, block size
// and launch gpu kernel
// each launch should transfer new frontier
void one_step_bottom_up_gpu() {
	// transfer current frontier and frontier_next to gpu
	cudaMemcpy(frontier_g, frontier, size_frontier_g, cudaMemcpyHostToDevice);
	cudaMemset(frontier_next_g, 0, size_frontier_next_g);
	// cudaMemcpy(frontier_next_g, frontier_next, size_frontier_next_g, cudaMemcpyHostToDevice);	


	// get suitable dim
	dim3 dimGrid;
	dim3 dimBlock;
	dim_do_nothing(&dimGrid, &dimBlock);

	// launch gpu kernel
	// it should compute frontier_next_g
	Context context = {
		rank, size
#ifdef SIZE_MUST_BE_A_POWER_OF_TWO
		, lgsize
#endif
		, g.nlocalverts
	};

	do_nothing<<<dimGrid, dimBlock>>>(rowstarts_g, column_g, frontier_g, frontier_next_g, pred_g, context);

	cudaMemcpy(frontier_next, frontier_next_g, size_frontier_next_g, cudaMemcpyDeviceToHost);
}

// 
// nvcc -arch=sm_37 -c bottom_up_gpu.cu -o bottom_up_gpu.o
// 
// GPU
// multiprocessor (SM?): 13
// global memory: 12G
// constant memroy: 64k (2^16)
// shared memory per block: 48k (3 * 2^14)
// register per block: 64k (2^16)
// warp size: 32
// max num thread per multiprocessor (SM?): 2048 (2^11, 2 block)
// max num thread per block: 1024 (2^10)
// 
// 
// x block (max inf)
// y thread per block (max 2^10)
// 
// around 12 int64_t per thread -> 2^7 register -> max 2^9 thread per block
// total 2^14 vertex
// local 2^13 vertex -> 2^13 thread -> at least 2^3 blocks
// 
// at least 2^4 blocks, each 2^9 thread -> 1x SM
// 
// lets try 2^6 blocks=64, each 2^7=128 thread
// 
// frontier bit map: 2^16 / 64 * 8 = 8k
// 
// 
// without considering global memory
// 
// 


// Device 3: "Tesla K80"
//   CUDA Driver Version / Runtime Version          7.5 / 7.5
//   CUDA Capability Major/Minor version number:    3.7
//   Total amount of global memory:                 12288 MBytes (12884705280 bytes)
//   (13) Multiprocessors, (192) CUDA Cores/MP:     2496 CUDA Cores
//   GPU Max Clock rate:                            824 MHz (0.82 GHz)
//   Memory Clock rate:                             2505 Mhz
//   Memory Bus Width:                              384-bit
//   L2 Cache Size:                                 1572864 bytes
//   Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
//   Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
//   Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
//   Total amount of constant memory:               65536 bytes
//   Total amount of shared memory per block:       49152 bytes
//   Total number of registers available per block: 65536
//   Warp size:                                     32
//   Maximum number of threads per multiprocessor:  2048
//   Maximum number of threads per block:           1024
//   Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
//   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
//   Maximum memory pitch:                          2147483647 bytes
//   Texture alignment:                             512 bytes
//   Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
//   Run time limit on kernels:                     No
//   Integrated GPU sharing Host Memory:            No
//   Support host page-locked memory mapping:       Yes
//   Alignment requirement for Surfaces:            Yes
//   Device has ECC support:                        Disabled
//   Device supports Unified Addressing (UVA):      Yes
//   Device PCI Domain ID / Bus ID / location ID:   0 / 133 / 0
//   Compute Mode:
//      < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
// > Peer access from Tesla K80 (GPU0) -> Tesla K80 (GPU1) : Yes
// > Peer access from Tesla K80 (GPU0) -> Tesla K80 (GPU2) : No
// > Peer access from Tesla K80 (GPU0) -> Tesla K80 (GPU3) : No
// > Peer access from Tesla K80 (GPU1) -> Tesla K80 (GPU0) : Yes
// > Peer access from Tesla K80 (GPU1) -> Tesla K80 (GPU2) : No
// > Peer access from Tesla K80 (GPU1) -> Tesla K80 (GPU3) : No
// > Peer access from Tesla K80 (GPU2) -> Tesla K80 (GPU0) : No
// > Peer access from Tesla K80 (GPU2) -> Tesla K80 (GPU1) : No
// > Peer access from Tesla K80 (GPU2) -> Tesla K80 (GPU3) : Yes
// > Peer access from Tesla K80 (GPU3) -> Tesla K80 (GPU0) : No
// > Peer access from Tesla K80 (GPU3) -> Tesla K80 (GPU1) : No
// > Peer access from Tesla K80 (GPU3) -> Tesla K80 (GPU2) : Yes

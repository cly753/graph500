CFLAGS_ = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math -DGRAPH_GENERATOR_MPI # -g -pg
# CFLAGS = -g -Wall -Drestrict= -DGRAPH_GENERATOR_MPI
LDFLAGS = -fopenmp -g -L.
# -g -pg
MPICC = mpicc

CFLAGS = ${CFLAGS_} -Isrc -Isrc/util -Isrc/onedim -I. -I..
# CFLAGS = ${CFLAGS_}

# all: graph500_mpi_simple graph500_mpi_one_sided graph500_mpi_replicated graph500_mpi_replicated_csc graph500_mpi_custom
all: graph500_mpi_onedim

HEADERS_ = common.h oned_csr.h oned_csc.h redistribute.h mpi_workarounds.h onesided.h
HEADERS = ${HEADERS_} src/constants.h src/util/print.h src/onedim/bfs.h src/onedim/top_down.h src/onedim/bottom_up.h src/onedim/frontier_tracker.h
# HEADERS = ${HEADERS_}

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
# GENERATOR_OBJECTS = $(patsubst %.c, %.o, $(GENERATOR_SOURCES))
SOURCES_ = main.c oned_csr.c oned_csc.c utils.c validate.c onesided.c onesided_emul.c
SOURCES = ${SOURCES_} src/util/print.c
# OBJECTS_ = $(patsubst %.c, %.o, $(SOURCES))

SOURCES_ONEDIM = src/onedim/onedim.c src/onedim/bfs.c src/onedim/top_down.c src/onedim/bottom_up.c src/onedim/frontier_tracker.c src/onedim/frontier_tracker_gpu.c
# OBJECTS = ${OBJECTS_} $(patsubst %.c, %.o, $(SOURCES_ONEDIM))
# SOURCES = ${SOURCES_}

graph500_mpi_simple: bfs_simple.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_simple bfs_simple.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_one_sided: bfs_one_sided.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_one_sided bfs_one_sided.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated: bfs_replicated.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated bfs_replicated.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_replicated_csc: bfs_replicated_csc.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_replicated_csc bfs_replicated_csc.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_custom: bfs_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_custom bfs_custom.c $(SOURCES) $(GENERATOR_SOURCES) -lm


NVCC       = /usr/local/cuda-7.5/bin/nvcc
CUDA_HOME  = /usr/local/cuda-7.5
LIB_OPT  += -L$(CUDA_HOME)/lib64 -lcudart -lstdc++
SUP_INCL += -I$(CUDA_HOME)/include

NV_CFLAGS_ = -Drestrict=__restrict__ -O3 $(SUP_INCL)
NV_CFLAGS = ${NV_CFLAGS_} -Isrc -Isrc/util -Isrc/onedim -I. -I.. -I/opt/openmpi-gcc/include
NV_LDFLAGS = -g
NV_ARCH = sm_37

HEADERS_ONEDIM_GPU = $(HEADERS)
SOURCES_ONEDIM_GPU = src/onedim/bottom_up_gpu_3.cu
OBJECT_ONEDIM_GPU = bottom_up_gpu_3.o

bottom_up_gpu: $(SOURCES_ONEDIM_GPU) $(HEADERS)
	$(NVCC) $(NV_CFLAGS) -arch=$(NV_ARCH) -c $(SOURCES_ONEDIM_GPU)

# mpi_objects: $(SOURCES) $(SOURCES_ONEDIM) $(HEADERS)
# 	$(MPICC) $(CFLAGS) -c $(SOURCES) $(SOURCES_ONEDIM)

# generator_objects: $(GENERATOR_SOURCES) $(HEADERS)
# 	$(MPICC) $(CFLAGS) -c $(GENERATOR_SOURCES)

# OBJECTS=$(wildcard *.o)
# graph500_mpi_onedim: mpi_objects generator_objects
# 	$(MPICC) $(LDFLAGS) $(OBJECTS) -lm -o graph500_mpi_onedim

graph500_mpi_onedim: $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) $(SOURCES_ONEDIM) bottom_up_gpu
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_onedim $(SOURCES_ONEDIM) $(SOURCES) $(GENERATOR_SOURCES) $(OBJECT_ONEDIM_GPU) -lm $(LIB_OPT)

clean:
	-rm -f graph500_mpi_* *.o *.a


# run command 
# /opt/openmpi-gcc/bin/mpirun -mca orte_base_help_aggregate 0 --allow-run-as-root -np 2 -x LD_LIBRARY_PATH ./run_g500.sh 3 1

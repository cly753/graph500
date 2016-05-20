CFLAGS_ = -fopenmp -g -Wall -Drestrict=__restrict__ -O3 -DNDEBUG -ffast-math -DGRAPH_GENERATOR_MPI # -g -pg
# CFLAGS = -g -Wall -Drestrict= -DGRAPH_GENERATOR_MPI
LDFLAGS = -fopenmp -g # -g -pg
MPICC = mpicc

CFLAGS = ${CFLAGS_} -Isrc -Isrc/util -Isrc/onedim -I. -I..
# CFLAGS = ${CFLAGS_}

# all: graph500_mpi_simple graph500_mpi_one_sided graph500_mpi_replicated graph500_mpi_replicated_csc graph500_mpi_custom
all: graph500_mpi_onedim

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES_ = main.c oned_csr.c oned_csc.c utils.c validate.c onesided.c onesided_emul.c
HEADERS_ = common.h oned_csr.h oned_csc.h redistribute.h mpi_workarounds.h onesided.h

HEADERS = ${HEADERS_} src/constants.h src/util/save.h src/util/print.h src/onedim/bfs.h
# HEADERS = ${HEADERS_}
SOURCES = ${SOURCES_} src/util/save.c src/util/print.c
SOURCES_ONEDIM = src/onedim/onedim.c src/onedim/bfs.c
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

graph500_mpi_onedim: $(SOURCES_ONEDIM) $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_mpi_onedim $(SOURCES_ONEDIM) $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f graph500_mpi_* *.o *.a

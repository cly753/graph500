## Graph500

### Algorithm

#### Version {reference}

- Asynchronous BFS
- 1-d partition.

##### Profiling

- ~~~ TBD 

#### Version CPU + GPU

- Level-synchronized BFS.
- Use 1-d partiton to distribute the graph.
- CSR for bottom up.
- CSC for top down.
- Bitmap tracking global frontier.
- Synchronize each level using All-to-all.
- Implementation is based on MPI.
- Switch to GPU at slowest level.

##### Profiling

- ~~~ TBD

### Code

Implementation goes in `graph500-2.1.4\mpi\src`.
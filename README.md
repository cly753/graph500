## Graph500

### Algorithm

#### Version {reference}

- Asynchronous BFS
- 1-d partition.

##### Profiling

- ~~~ TBD 

#### Version x

- Level-synchronized BFS.
- Use 1-d partiton to distribute the graph.
- Convert CSR to CSC.
- Synchronize each level using All-to-all.
- Implementation is based on MPI.

##### Profiling

- ~~~ TBD 

#### Version y

- Level-synchronized bottom-up BFS.
- Use CSR.
- Bitmap tracking global frontier.
- Filter duplicated edges.

##### Profiling

- ~~~ TBD

#### Version z

- Level-synchronized top-down + bottom-up BFS.
- Use CSR and construct CSC-like array.
- Bitmap tracking global frontier.
- Filter duplicated edges.

##### Profiling

- ~~~ TBD

#### Version pure-gpu (current)

- Level-synchronized bottom-up BFS.
- Use CSR and recover whole edge array.
- Bitmap tracking global frontier.
- Filter duplicated edges.
- All computations are done in GPU.
- Each GPU thread process one edge.
- If cuda-aware support is available, should be much faster.

##### Profiling

- ~~~ TBD

### Code

Implementation goes in `graph500-2.1.4\mpi\src`.
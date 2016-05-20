## Graph500

### Algorithm

#### Version x

- Level-synchronized BFS.
- Use 1-d partiton to distribute the graph.
- Synchronize each level using All-to-all.
- Implementation is based on MPI.

### Code

Our implementation goes in `graph500-2.1.4\mpi\src`.
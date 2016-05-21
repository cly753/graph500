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
- Synchronize each level using All-to-all.
- Implementation is based on MPI.

##### Profiling

- ~~~ TBD 

#### Version y

- Level-synchronized bottom-up BFS.
- Bitmap tracking global frontier.

##### Profiling

- ~~~ TBD

### Code

Implementation goes in `graph500-2.1.4\mpi\src`.
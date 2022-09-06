# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

In the sixth attempt, we focus on the new communication strategy. There is no dependence on "shift" and "incidence" arrays, and the communication scheme is much more clear. This should also fix the deadlocking problem, and allow easier benchmarking.

In addition to the improved bulk communication, column-reindexing has also been greatly improved. Specifically, the hash-map solution of att5 has been replaced with a couple of for loops. To do this improvement, we had to de-couple communication with the order of the heri-elements within the communication buffer. The previous comm strategy has heavily dependant on this ordering. That is why these two important improvements had to happen in tandem.

Moving forward, the plans are: to further improve communication (via. computation/communication overlap), implement a sigma-scope "reordering", and to fully implement the ell/crs formats for comparison.
# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

In this fourth attempt, we've achieved our first "working" halo element communication scheme for the local x vectors. This, along with the other advances, has given us a rough implementation of MPI+X SPMVM for the SELL-C-sigma data storage format. The support for crs, ell, and coo are not fully implemented as of yet. Even with scs, seg faults and double free'd errors are plentiful at the moment. But evenso, the code does what it is supposed to do in some cases.

Att3 using a different starting codebase did not work as planned, and so with this fourth attempt, we've pressed on with the original codebase. There have been major modifications to organization, and more effort has been made to modularize ideas into functions and headers.

Optionally, we collect the proc-local results to every processes, essentailly reconstructing the entire result vector on each process. This is used used for validation against Intel's MKL library.

Moving forward, the plans are twofold. First, implement the following better and use it for benchmarking different kernels on different hardware. Then, use the accelerated SPMVM in existing frameworks/solvers.

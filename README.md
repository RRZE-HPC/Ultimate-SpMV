# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

For the fifth attempt, the focus has been on restructuring/reorganizing of the code. Also, a nonblocking communication scheme is to be implemented, with proc-local buffers communicated instead of individual elements, as was the case in previous iterations of the code. Accuracy of results from att4 is to be sustained. 

The sparse matrix storage formats crs, ell, and coo will have their first implementations, along side scs. The mtx struct, and proc-local x and y are initialized outside of bench_spmv(), so that all sotrage formats can use the same code.

There is now two operational modes, "bench" and "solve", selected in cli by the user:

With the "bench" mode, the main COMM-SPMVM-SWAP loop will run as many times as is needed, i.e. for a suitable amount of time, in order to obtain accurate benchmark results. In this mode, x and y are not swapped and then multiplied, but rather, swapped with a dummy vector in order to obtain a realistic performace. The number of iterations of the main COMM-SPMVM-SWAP loop is determined at runtime.

In "solve" mode, the proc-local x and y vectors are in fact swapped to provide an accurate result, after the user specified number of iterations. Only in solve mode does the user have the option to validate the result, as "bench" mode will not provide a true result for the given number of iterations. The number of iterations of the main COMM-SPMVM-SWAP loop is determined by the user as a cli arguement. Depending on the matrix used, after arpund 5-15 revisions, there is overflow in some elements.

The plans for the next iteration of the code (att6) are... TBD.
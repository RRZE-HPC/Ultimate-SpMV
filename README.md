# Ultimate-SpMV
MPI+X SpMV with SELL-C-sigma format

For the fifth attempt, the focus has been on restructuring/reorganizing of the code. 

The sparse matrix storage formats crs and ell are implemented as special cases of scs. The proc local struct are initialized outside of bench_spmv(), so that all storage formats can use the same code.

There is now two operational modes, "bench" and "solve", selected in cli by the user:

With the "bench" mode, the main COMM-SPMVM-SWAP loop will run as many times as is needed, i.e. for a suitable amount of time, in order to obtain accurate benchmark results. In this mode, x and y are not swapped and then multiplied, but rather, swapped with a dummy vector in order to obtain a realistic performace. The number of iterations of the main COMM-SPMVM-SWAP loop is determined at runtime. This mode is currently flawed, and results in deadlocks "randomly". The cause of this is definetly due to the naive communication strategy.

In "solve" mode, the proc-local x and y vectors are in fact swapped to provide an accurate result, after the user specified number of iterations. Only in solve mode does the user have the option to validate the result, as "bench" mode will not provide a true result for the given number of iterations. The number of iterations of the main COMM-SPMVM-SWAP loop is determined by the user as a cli arguement. Depending on the matrix used, after around 5-15 revisions, there is overflow in some elements in the author's experience.

The plans for the next iteration of the code (att6) are to\ implement a bulk communication scheme, with proc-local buffers communicated instead of individual elements (as was the case in previous iterations of the code). This should aid in prevention of deadlocks, cleaner code, and better bandwidth utilization.
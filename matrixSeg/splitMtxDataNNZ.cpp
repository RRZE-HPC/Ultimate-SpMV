    // Method 2. Segment by number of non-zero elements
    // int nnzPerProc = mtx.nnz / commSize;
    // int nnzRemainder = mtx.nnz % nnzPerProc;

    // // Still need index of nnz
    // for(int nz = myRank * nnzPerProc; nz < (myRank + 1) * nnzPerProc; ++nz){
    //     // I dont think this idea works here
    //     getIndex<IT>(mtx.values, nz)

        //there needs to be some flag that states:
        // if nnzPerProc is reached, just give rest of row to processes

    // }
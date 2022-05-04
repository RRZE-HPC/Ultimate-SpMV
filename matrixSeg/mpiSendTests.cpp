#include <iostream>
#include <mpi.h>

// just send struct of ints

// struct Node{
//     int x;
//     int y;
//     int z;
// };

// int main(int argc, char* argv[]){
//     int myRank, commSize;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &commSize);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
//     MPI_Status status;

//     // Create Datatype
//     MPI_Datatype nodeWithInts;
//     MPI_Type_contiguous(3, MPI_INT, &nodeWithInts);
//     MPI_Type_commit(&nodeWithInts);

//     Node node1;

//     if(myRank == 0){
//         node1.x = 1;
//         node1.y = 6;
//         node1.z = 99;

//         // NOTE: only sending one of these
//         MPI_Send(&node1, 1, nodeWithInts, 1, 42, MPI_COMM_WORLD);
//     }
//     else if(myRank == 1){
//         Node recvdNode;
//         MPI_Recv(&recvdNode, 1, nodeWithInts, 0, 42, MPI_COMM_WORLD, &status);

//         // printf("Status: %d\n", status.MPI_ERROR);

//         printf("%i, %i, %i\n", recvdNode.x, recvdNode.y, recvdNode.z);

//     }

//     MPI_Type_free(&nodeWithInts);
// }

// send struct with ints and floats
// struct Node{
//     int loc[3];
//     float grey;
//     float hue;
// } nodeSend, nodeRecv;

// int main(int argc, char* argv[]){
//     int myRank, commSize;
//     int blockLengthArr[2];
//     MPI_Aint displacementArr[3], firstAddress, secondAddress, thirdAddress;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD, &commSize);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
//     MPI_Status status;

//     // Create Datatype
//     MPI_Datatype typeArr[2], sendRecvType;
//     typeArr[0] = MPI_INT;
//     typeArr[1] = MPI_FLOAT;
//     blockLengthArr[0] = 3; // using 3 int elements
//     blockLengthArr[1] = 2; // using 2 float elements
//     MPI_Get_address(&nodeSend.loc, &firstAddress); // get addresses of "nodeSend" members
//     MPI_Get_address(&nodeSend.grey, &secondAddress);
//     MPI_Get_address(&nodeSend.hue, &thirdAddress);
//     displacementArr[0] = (MPI_Aint) 0; // calculate displacements
//     displacementArr[1] = MPI_Aint_diff(secondAddress, firstAddress);
//     displacementArr[2] = MPI_Aint_diff(thirdAddress, firstAddress);
//     MPI_Type_create_struct(2, blockLengthArr, displacementArr, typeArr, &sendRecvType);
//     MPI_Type_commit(&sendRecvType);


//     if(myRank == 0){
//         nodeSend.loc[0] = 33; nodeSend.loc[1] = 0; nodeSend.loc[2] = 62; 
//         nodeSend.grey = 6.33;
//         nodeSend.hue = .545;

//         // NOTE: only sending one of these
//         MPI_Send(&nodeSend, 1, sendRecvType, 1, 42, MPI_COMM_WORLD);
//     }
//     else if(myRank == 1){
//         MPI_Recv(&nodeRecv, 1, sendRecvType, 0, 42, MPI_COMM_WORLD, &status);

//         // printf("Status: %d\n", status.MPI_ERROR);

//         printf("[%i, %i, %i], (%f, %f)\n", nodeRecv.loc[0], nodeRecv.loc[1], nodeRecv.loc[2], nodeRecv.grey, nodeRecv.hue);

//     }

//     MPI_Type_free(&sendRecvType);
// }

// send struct with ints and bools
struct Node{
    int loc[3];
    bool isActive;
} nodeSend, nodeRecv;

int main(int argc, char* argv[]){
    int myRank, commSize;
    int blockLengthArr[2];
    MPI_Aint displacementArr[2], firstAddress, secondAddress;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Status status;

    // Create Datatype
    MPI_Datatype typeArr[2], sendRecvType;
    typeArr[0] = MPI_INT;
    typeArr[1] = MPI_CXX_BOOL;
    blockLengthArr[0] = 3; // using 3 int elements
    blockLengthArr[1] = 1; // using 2 float elements
    MPI_Get_address(&nodeSend.loc, &firstAddress); // get addresses of "nodeSend" members
    MPI_Get_address(&nodeSend.isActive, &secondAddress);
    displacementArr[0] = (MPI_Aint) 0; // calculate displacements
    displacementArr[1] = MPI_Aint_diff(secondAddress, firstAddress);
    MPI_Type_create_struct(2, blockLengthArr, displacementArr, typeArr, &sendRecvType);
    MPI_Type_commit(&sendRecvType);


    if(myRank == 0){
        nodeSend.loc[0] = 33; nodeSend.loc[1] = 0; nodeSend.loc[2] = 62; 
        nodeSend.isActive = true;

        // NOTE: only sending one of these
        MPI_Send(&nodeSend, 1, sendRecvType, 1, 42, MPI_COMM_WORLD);
    }
    else if(myRank == 1){
        MPI_Recv(&nodeRecv, 1, sendRecvType, 0, 42, MPI_COMM_WORLD, &status);

        // printf("Status: %d\n", status.MPI_ERROR);

        printf("[%i, %i, %i], %d\n", nodeRecv.loc[0], nodeRecv.loc[1], nodeRecv.loc[2], nodeRecv.isActive);

    }

    MPI_Type_free(&sendRecvType);
}

// send struct with vectors
struct Node{
    std::vector<double> loc;
    std::vector<int> color;
};

int main(int argc, char* argv[]){
    int myRank, commSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Status status;

    // Create Datatype
    MPI_Datatype nodeWithInts;
    MPI_Type_contiguous(3, MPI_INT, &nodeWithInts);
    MPI_Type_commit(&nodeWithInts);

    Node node1;

    if(myRank == 0){
        node1.x = 1;
        node1.y = 6;
        node1.z = 99;

        // NOTE: only sending one of these
        MPI_Send(&node1, 1, nodeWithInts, 1, 42, MPI_COMM_WORLD);
    }
    else if(myRank == 1){
        Node recvdNode;
        MPI_Recv(&recvdNode, 1, nodeWithInts, 0, 42, MPI_COMM_WORLD, &status);

        // printf("Status: %d\n", status.MPI_ERROR);

        printf("%i, %i, %i\n", recvdNode.x, recvdNode.y, recvdNode.z);

    }

    MPI_Type_free(&nodeWithInts);
}

// combine all three
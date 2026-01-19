#include <iostream>
#include <mpi.h>
#include <time.h>

#include "matrixManager.h"

#define ROOT_RANK 0

using namespace std;

//---------------------------------------//
//Initialize vector for multiplication with random values from 1 to 100
void initVec(vector<double> &v, int cols){
	for(int i=0; i<cols; i++){
		v[i] = rand() % 100 + 1;
	}
}

void initMatrix(string matrixName, MatrixCOO &coo, MatrixCSR &csr) {

    coo.readMatrixFromFile(matrixName);
    csr.convertCOOInCSR(coo); 

	vector<double> vec(coo.cols, 0); 
	
    //print coo and csr
    coo.print();
    csr.print();

    //initialize the entire multiply vector
    initVec(vec, coo.cols);

    cout << "Multiply vector: ";
    for(auto i: vec){
        cout << i << " ";
    } cout << endl;

}
/*
void spmv(const CSR& csr, const vector<double> &v, vector<double> &r){
    int rows = (int)r.size();

    for(int ip=0; ip<rows; ip++){		
    double sum = 0.0;
        for (int k = csr.pointer[ip]; k < csr.pointer[ip+1]; k++) {
            sum += csr.data[k] * v[ csr.index[k] ];
        }
        r[ip] = sum;
	}
}
*/



int main(int argc, char** argv) {
	
	srand(time(0));

    int rank, world_size;
    int total_rows; //total number of rows
    int total_nnz; //total number of nnz
    int local_nrows; //number of rows that each process manages
    int local_nnz; //used to resize the local arrays of each process

    MatrixCOO coo; //populated by only the ROOT rank
    MatrixCSR csr; //populated by only the ROOT rank

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    vector<int> nnz_counts(world_size, 0); //contains the number of elements to send to each process
    vector<int> nnz_displs(world_size, 0); //contains the displacement to apply to the message sent to each process.

    vector<int> shuffled_lengths; //csr row pointer array grouped by rank
    vector<int> shuffled_idx; //csr index array grouped by rank
    vector<double> shuffled_data; //csr data array grouped by rank
    
    vector<int> local_lengths;
    vector<int> local_idx;
    vector<double> local_data;
    

	if (argc < 1) {
        cout << "Usage: ./distributed_spmv <matrixName>" << endl;
        return -1;
    }

    string matrixName = argv[1];
    
    if(rank==ROOT_RANK){
        initMatrix(matrixName, coo, csr);

        total_rows = coo.rows; 
        total_nnz = coo.nnz;

        printf("[MPI process %d] I am the broadcast root, and send value %d.\n", rank, total_rows);
    } 

    MPI_Bcast(&total_rows, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_rows

    if(rank!=ROOT_RANK){
        printf("[MPI process %d] I am a broadcast receiver, and obtained value %d.\n", rank, total_rows);
    }

    //Every rank calculates its number of local rows
    local_nrows = 0;
    for (int i = 0; i < total_rows; i++) {
        if (i % world_size == rank) local_nrows++;
    }
   
    // Rank 0 calculates the counts and displacements for Scatterv
    if (rank == ROOT_RANK) {
        for (int i = 0; i < total_rows; i++) {
            int target = i % world_size;
            int row_nnz = csr.pointer[i+1] - csr.pointer[i];
            nnz_counts[target] += row_nnz;
        }
        
        // Calculate displacements (offsets)
        for (int i = 1; i < world_size; i++) {
            nnz_displs[i] = nnz_displs[i-1] + nnz_counts[i-1];
        }
    }


    vector<int> current_len_pos(world_size, 0);
    vector<int> len_counts(world_size);
    vector<int> len_displs(world_size, 0);

    for(int i=0; i<world_size; i++) {
        len_counts[i] = (total_rows / world_size) + (i < (total_rows % world_size) ? 1 : 0);
        if(i > 0) len_displs[i] = len_displs[i-1] + len_counts[i-1];
    }

    //Rank 0 organizes the full_idx and full_val so they are grouped by rank.
    if (rank == ROOT_RANK) {
        shuffled_idx.resize(total_nnz);
        shuffled_data.resize(total_nnz);
        shuffled_lengths.resize(total_rows);
        
        // Track where we are writing for each rank within the shuffled buffers
        vector<int> current_pos = nnz_displs; 
        vector<int> write_ptr = len_displs;

        for(int i=0; i<world_size; i++) {
            len_counts[i] = (total_rows / world_size) + (i < (total_rows % world_size) ? 1 : 0);
            if(i > 0) len_displs[i] = len_displs[i-1] + len_counts[i-1];
        }

        for (int i = 0; i < total_rows; i++) {
            int target = i % world_size;
            int row_start = csr.pointer[i];
            int row_nnz = csr.pointer[i+1] - csr.pointer[i];

            // Copy indices and values into the rank's assigned block
            copy(csr.index.begin() + row_start, 
                csr.index.begin() + row_start + row_nnz, 
                shuffled_idx.begin() + current_pos[target]);
            
            copy(csr.data.begin() + row_start, 
                csr.data.begin() + row_start + row_nnz, 
                shuffled_data.begin() + current_pos[target]);

            current_pos[target] += row_nnz;

            shuffled_lengths[write_ptr[target]] = row_nnz;
            write_ptr[target]++;
        }
    }

    // Every rank needs to know its own NNZ count to resize its local arrays
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    local_idx.resize(local_nnz);
    local_data.resize(local_nnz);
    local_lengths.resize(local_nrows);

    MPI_Scatterv(shuffled_idx.data(), nnz_counts.data(), nnz_displs.data(), MPI_INT,
                local_idx.data(), local_nnz, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled_data.data(), nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
             local_data.data(), local_nnz, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT,
             local_lengths.data(), local_nrows, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

        
    // Rebuild the local CSR pointer array
    vector<int> local_ptr(local_nrows + 1);
    local_ptr[0] = 0;
    for (int i = 0; i < local_nrows; i++) {
        local_ptr[i+1] = local_ptr[i] + local_lengths[i];
    }

    // print received local_idx and local_data
    for (int r = 0; r < world_size; r++) {
        if (rank == r) {
            printf("\n--- Rank %d | local_nrows: %d | local_nnz: %d ---\n", 
                    rank, local_nrows, (int)local_idx.size());
            
            printf("local_pointer: [ ");
            for (int val : local_ptr) printf("%d ", val);
            printf("]\n");

            printf("local_idx: [ ");
            for (int val : local_idx) printf("%d ", val);
            printf("]\n");

            printf("local_val: [ ");
            for (double val : local_data) printf("%.2f ", val);
            printf("]\n");
            
            // Push the output to the console immediately
            fflush(stdout); 
        }
        // Wait for the current rank to finish printing before the next one starts
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // --------------------------------------- //




    MPI_Finalize();
  
	return 0;
}


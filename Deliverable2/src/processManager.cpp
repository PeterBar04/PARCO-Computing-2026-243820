#include "processManager.h"

// Metadata for Scatterv (Counts and Displacements)
struct CommMetadata {
    vector<int> nnz_counts;
    vector<int> nnz_displs;
    vector<int> len_counts;
    vector<int> len_displs;

    // Helper to resize all vectors at once
    void init(int worldSize) {
        nnz_counts.assign(worldSize, 0);
        nnz_displs.assign(worldSize, 0);
        len_counts.assign(worldSize, 0);
        len_displs.assign(worldSize, 0);
    }
};

// Buffers for Shuffling (Only Root will fill these)
struct ShuffledBuffers {
    vector<int> lengths;
    vector<int> indices;
    vector<double> data;

    void allocate(int total_rows, int total_nnz) {
        lengths.assign(total_rows, 0);
        indices.assign(total_nnz, 0);
        data.assign(total_nnz, 0.0);
    }
};


Process::Process(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        localCSR.nrows = 0;
    }

Process::~Process() {
    MPI_Finalize();
}

void Process::setupData(string matrixName){
    if(rank==ROOT_RANK){
        
        globalCoo.readMatrixFromFile(matrixName);
        globalCSR.convertCOOInCSR(globalCoo);
        
        total_rows = globalCoo.rows; 
        total_nnz = globalCoo.nnz;

        globalX.resize(globalCoo.cols);
        //initialize the entire multiply vector
        //initVec(globalX, coo.cols);

        globalCSR.print();
        cout << "Multiply vector: ";
        for(auto i: globalX){
            cout << i << " ";
        } cout << endl;

        //printf("[MPI process %d] I am the broadcast root, and send value %d.\n", proc.rank, total_rows);
    }

    MPI_Bcast(&total_rows, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_rows
    MPI_Bcast(&total_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_nnz

    /*
    if(proc.rank!=ROOT_RANK){
        printf("[MPI process %d] I am a broadcast receiver, and obtained value %d.\n", proc.rank, total_rows);
    }
    */

}

void Process::scatterMatrix() {

    // Every rank needs metadata to participate in MPI_Scatterv
    CommMetadata meta;
    ShuffledBuffers shuffled;
    meta.init(worldSize);

    // buffer that stores lenght of rows
    vector<int> local_lengths;

    calculateLocalRows();
    calculateCountDispl(meta);
    
    if (rank == ROOT_RANK) {
        shuffled.allocate(total_rows, total_nnz);
        //Rank 0 organizes the full_idx and full_val so they are grouped by rank.
        shuffleVectors(meta, shuffled);
    }
    
    scattering(meta, shuffled, local_lengths);

    buildCSRpointer(local_lengths);
}


void Process::calculateLocalRows(){
    //Every rank calculates its number of local rows
    for (int i = 0; i < total_rows; i++) {
        if (i % worldSize == rank) localCSR.nrows++;
    }
}

void Process::calculateCountDispl(CommMetadata &meta){
    // Calculate row length counts/displ (len_counts) for ALL ranks
    for(int i=0; i<worldSize; i++) {
        meta.len_counts[i] = (total_rows / worldSize) + (i < (total_rows % worldSize) ? 1 : 0);
        if(i > 0) meta.len_displs[i] = meta.len_displs[i-1] + meta.len_counts[i-1];
    }

    // Calculate NNZ counts/displ (ONLY ROOT needs to loop through the matrix)
    if(rank == ROOT_RANK){
        for (int i = 0; i < total_rows; i++) {
            int target = i % worldSize;
            int nnzInRow = globalCSR.pointer[i+1] - globalCSR.pointer[i];
            meta.nnz_counts[target] += nnzInRow;
        }
        
        // Calculate displacements (offsets)
        for (int i = 1; i < worldSize; i++) {
            meta.nnz_displs[i] = meta.nnz_displs[i-1] + meta.nnz_counts[i-1];
        }
    }
    
}

void Process::shuffleVectors(CommMetadata &meta, ShuffledBuffers &shuffled){

    // Track where we are writing for each rank within the shuffled buffers
    vector<int> current_pos = meta.nnz_displs; 
    vector<int> write_ptr = meta.len_displs;

    for (int i = 0; i < total_rows; i++) {
        int target = i % worldSize;
        int row_start = globalCSR.pointer[i];
        int nnzInRow = globalCSR.pointer[i+1] - globalCSR.pointer[i];

        // Copy indices and values into the rank's assigned block
        copy(globalCSR.index.begin() + row_start, 
            globalCSR.index.begin() + row_start + nnzInRow, 
            shuffled.indices.begin() + current_pos[target]);
        
        copy(globalCSR.data.begin() + row_start, 
            globalCSR.data.begin() + row_start + nnzInRow, 
            shuffled.data.begin() + current_pos[target]);

        current_pos[target] += nnzInRow;

        shuffled.lengths[write_ptr[target]] = nnzInRow;
        write_ptr[target]++;
    }
}

void Process::scattering(CommMetadata &meta, ShuffledBuffers &shuffled, vector<int> &local_lengths) {

    // Every rank needs to know its own NNZ count to resize its local arrays
    MPI_Scatter(meta.nnz_counts.data(), 1, MPI_INT, &localCSR.nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    localCSR.index.resize(localCSR.nnz);
    localCSR.data.resize(localCSR.nnz);
    local_lengths.resize(localCSR.nrows);

    MPI_Scatterv(shuffled.indices.data(), meta.nnz_counts.data(), meta.nnz_displs.data(), MPI_INT,
                localCSR.index.data(), localCSR.nnz, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.data.data(), meta.nnz_counts.data(), meta.nnz_displs.data(), MPI_DOUBLE,
             localCSR.data.data(), localCSR.nnz, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.lengths.data(), meta.len_counts.data(), meta.len_displs.data(), MPI_INT,
             local_lengths.data(), localCSR.nrows, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

}

// Rebuild the local CSR pointer array
void Process::buildCSRpointer(vector<int> &local_lengths){
    localCSR.pointer.assign(localCSR.nrows + 1, 0);
    for (int i = 0; i < localCSR.nrows; i++) {
        localCSR.pointer[i+1] = localCSR.pointer[i] + local_lengths[i];
    }
}

void Process::print(){
    // print received local_idx and local_data
    for (int r = 0; r < worldSize; r++) {
        if (rank == r) {
            printf("\n--- Rank %d | local_nrows: %d | local_nnz: %d ---\n", 
                    rank, localCSR.nrows, (int)localCSR.index.size());
            
            printf("local_pointer: [ ");
            for (int val : localCSR.pointer) printf("%d ", val);
            printf("]\n");

            printf("local_idx: [ ");
            for (int val : localCSR.index) printf("%d ", val);
            printf("]\n");

            printf("local_val: [ ");
            for (double val : localCSR.data) printf("%.2f ", val);
            printf("]\n");
            
            // Push the output to the console immediately
            fflush(stdout); 
        }
        // Wait for the current rank to finish printing before the next one starts
        MPI_Barrier(MPI_COMM_WORLD);
    }
}



//Initialize vector for multiplication with random values from 1 to 100
void initVec(vector<double> &v, int cols){
	for(int i=0; i<cols; i++){
		v[i] = rand() % 100 + 1;
	}
}

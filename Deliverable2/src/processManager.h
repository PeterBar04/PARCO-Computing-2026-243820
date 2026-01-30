#ifndef PROCESSMANAGER_H
#define PROCESSMANAGER_H

#include "matrixManager.h"
#include <mpi.h>
#include <random>
#include <set>
#include <unordered_map> 
#include <algorithm>

using namespace std;

#define ROOT_RANK 0

struct CommMetadata;
struct ShuffledBuffers;

class Process {
private:
    int rank;       // rank ID
    int worldSize;  // total number of ranks

    // Global Metadata (All ranks should know these)
    int total_rows;
    int total_nnz;
    // --------------- //

    MatrixCOO globalCoo; //populated by only the ROOT rank
    MatrixCSR globalCSR; //populated by only the ROOT rank
    
    vector<double> globalX; //vector used for multiplication, only ROOT rank knows it entirely
    vector<double> localX; //local vector used for multiplication

    MatrixCSR localCSR; // The actual workload for each process

    vector<double> localY; // Local result of multiplication
    
    // Setup containers for communication of ghost values
    vector<vector<int>> requests;
    vector<int> send_counts, recv_counts;
    vector<int> send_displs, recv_displs;

    // Values used to store values of x exchanged with other ranks
    unordered_map<int, int> ghost_map;
    vector<double> ghost_buffer;

    // Renumbered Indices for fast math
    // This replaces localCSR.index for the actual calculation
    vector<int> renumbered_indices;

    // A unified vector: [ Local X values | Ghost X values ]
    // This allows us to access ANY value (local or ghost) using a simple array index.
    vector<double> combinedX;

    //Persistent Communication Buffers
    // We calculate these ONCE, so we don't need to ask "what do you need?" every iteration.
    vector<int> indices_to_send; // Local indices of data I need to pack and send
  

    // Time values
    double comm_time;
    double comp_time;

    void initGlobalX(vector<double>& v, int cols);

    // Methods used in scatterData()
    void calculateLocalRowsCols(CommMetadata &meta_x);
    void calculateCountDispl(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x);
    void shuffleVectors(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x, ShuffledBuffers &shuffled);
    void scattering(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x, ShuffledBuffers &shuffled, vector<int> &local_lengths);
    void buildCSRpointer(vector<int> &local_lengths);

    // Methods used in exchangeGhostIdentifier()
    void identifyGhostEntries(vector<vector<int>>& requests, unordered_map<int, int>& ghost_map);    
    void exchangeMetadata(const vector<vector<int>>& requests, 
                               vector<int>& num_indices_I_request, 
                               vector<int>& num_indices_others_need_from_me);
    
    // Methods used inside runCalculation()
    void exchangeGhostValues(); // Exchange values of vector x between ranks (do in loop)
    void computelocalSpMV(); // Compute spmv
    void calculateP90(vector<double> &local_comm, vector<double> &local_comp);  // Calculate p90 of times
    
    //Debug
    void printDebugIdentify(vector<vector<int>> requests);
    void printDebugExchangeMeta(const vector<int>& num_indices_I_request, const vector<int>& num_indices_others_need_from_me);
    void printDebugExchangeResult();

    

 public:
    // Constructor
    Process(int argc, char** argv); 

    //Getter
    int getRank();
    double getCommTime();
    double getCompTime();

    // Read the matrix from file and generate the multiply vector
    void setupData(string matrixName);

    // This method handles the "heavy lifting" of moving data from Root to others
    void scatterData();

    // Exchange identifier of vector x between ranks (made only once)
    void exchangeGhostIdentifier(); 

    void prepareOptimizedStructures();

    // Perform spmv in local
    void runCalculation(int num_iter);

    // Debug
    void print(); 
    void printMetrics();
  
    // Destructor
    ~Process(); 
};


#endif

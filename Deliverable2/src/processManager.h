#ifndef PROCESSMANAGER_H
#define PROCESSMANAGER_H

#include "matrixManager.h"
#include <mpi.h>
#include <random>
#include <set>
#include <map>
#include <algorithm>

using namespace std;

#define ROOT_RANK 0

struct CommMetadata;
struct ShuffledBuffers;

class Process {
private:
    int rank;
    int worldSize;

    // Global Metadata (All ranks should know these)
    int total_rows;
    int total_nnz;

    MatrixCOO globalCoo; //populated by only the ROOT rank
    MatrixCSR globalCSR; //populated by only the ROOT rank
    
    vector<double> globalX; //vector used for multiplication, only ROOT rank knows it entirely
    vector<double> localX; //local vector used for multiplication

    // The actual workload for each process
    MatrixCSR localCSR;
    

    void initGlobalX(vector<double>& v, int cols);
    void calculateLocalRows();
    void calculateCountDispl(CommMetadata &meta);
    void shuffleVectors(CommMetadata &meta, ShuffledBuffers &shuffled);
    void scattering(CommMetadata &meta, ShuffledBuffers &shuffled, vector<int> &local_lengths);
    void buildCSRpointer(vector<int> &local_lengths);

    void identifyGhostEntries(vector<vector<int>>& requests, map<int, int>& ghost_map);    
    void exchangeMetadata(const vector<vector<int>>& requests, 
                               vector<int>& num_indices_I_request, 
                               vector<int>& num_indices_others_need_from_me);
    void exchangeGhostValues(const vector<vector<int>>& requests, 
                                  const vector<int>& num_indices_I_request, 
                                  const vector<int>& num_indices_others_need_from_me,
                                  vector<double>& ghost_buffer);
    void performLocalSpMV(const map<int, int>& ghost_map, 
                               const vector<double>& ghost_buffer, 
                               vector<double>& y_local);

public:
    Process(int argc, char** argv);

    // This method would encapsulate all your logic
    void setupData(string matrixName);

    // This method handles the "heavy lifting" of moving data from Root to others
    void scatterMatrix();

    void distributeVectorX();
    void scatterVectorX();

    void performSpMV(const vector<double>& x, vector<double>& y_local);

    void print();
    
    ~Process();
};


#endif

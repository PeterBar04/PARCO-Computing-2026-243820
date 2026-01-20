#ifndef PROCESSMANAGER_H
#define PROCESSMANAGER_H

#include <mpi.h>
#include "matrixManager.h"

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
    
    vector<int> globalX; //vector used for multiplication, only ROOT rank knows it entirely
    
    // The actual workload for each process
    MatrixCSR localCSR;
    vector<int> localX;

    void calculateLocalRows();
    void calculateCountDispl(CommMetadata &meta);
    void shuffleVectors(CommMetadata &meta, ShuffledBuffers &shuffled);
    void scattering(CommMetadata &meta, ShuffledBuffers &shuffled, vector<int> &local_lengths);
    void buildCSRpointer(vector<int> &local_lengths);

public:
    Process(int argc, char** argv);

    // This method would encapsulate all your logic
    void setupData(string matrixName);

    // This method handles the "heavy lifting" of moving data from Root to others
    void scatterMatrix();

    void performSpMV(const vector<double>& x, vector<double>& y_local);

    void print();
    
    ~Process();
};


#endif

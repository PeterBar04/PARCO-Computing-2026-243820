    #include "processManager.h"

// Metadata for Scatterv (Counts and Displacements)
struct CommMetadata {
    vector<int> counts;
    vector<int> displs;

    // Helper to resize all vectors at once
    void init(int worldSize) {
        counts.assign(worldSize, 0);
        displs.assign(worldSize, 0);
    }
};

// Buffers for Shuffling (Only Root will fill these)
struct ShuffledBuffers {
    vector<int> lengths;
    vector<int> indices;
    vector<double> data;
    vector<double> x;

    void allocate(int total_rows, int total_nnz) {
        lengths.assign(total_rows, 0);
        indices.assign(total_nnz, 0);
        data.assign(total_nnz, 0.0);
        x.assign(total_rows, 0.0);
    }
};

// ----------------------------------------------------------------- //
// Constructor and destructor

Process::Process(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        localCSR.nrows = 0;

        total_rows = 0;
        total_nnz = 0;
    }

Process::~Process() {
    MPI_Finalize();
}

// ----------------------------------------------------------------- //
// PUBLIC METHODS

void Process::setupData(string matrixName){
    if(rank==ROOT_RANK){
        
        globalCoo.readMatrixFromFile(matrixName);
        globalCSR.convertCOOInCSR(globalCoo);
        
        total_rows = globalCoo.rows; 
        total_nnz = globalCoo.nnz;

        globalX.resize(globalCoo.cols);

        //Initialize vector for multiplication with random values from 1 to 100
        initGlobalX(globalX, globalCoo.cols); 
    }
    MPI_Bcast(&total_rows, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_rows
    MPI_Bcast(&total_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_nnz
}

void Process::scatterData() {

    // Every rank needs metadata to participate in MPI_Scatterv
    CommMetadata meta_nnz;
    CommMetadata meta_len;
    CommMetadata meta_x;

    meta_nnz.init(worldSize);
    meta_len.init(worldSize);
    meta_x.init(worldSize);

    ShuffledBuffers shuffled;
    
    // buffer that stores lenght of rows
    vector<int> local_lengths;

    // Every rank calculates its number of local rows and coloums
    calculateLocalRowsCols(meta_x);

    // Calculate own many data send (counts) to each rank and
    // the displacement to apply to the message sent to each process
    calculateCountDispl(meta_nnz, meta_len, meta_x);
    
    if (rank == ROOT_RANK) {
        shuffled.allocate(total_rows, total_nnz);
        //Rank 0 organizes the full_idx, full_val and full_lenghts so they are grouped by rank.
        shuffleVectors(meta_nnz, meta_len, meta_x, shuffled);
    }
    
    // Send chunks of the matrix and vector X to each rank
    scattering(meta_nnz, meta_len, meta_x, shuffled, local_lengths);

    // Build local CSR.pointer in each rank
    buildCSRpointer(local_lengths);
}

void Process::exchangeVectorX(){

    // Setup containers
    vector<vector<int>> requests;
    vector<int> send_counts, recv_counts;

    // Run the phases
    identifyGhostEntries(requests, this->ghost_map);

    exchangeMetadata(requests, send_counts, recv_counts);

    exchangeGhostValues(requests, send_counts, recv_counts, this->ghost_buffer);

    // --- DEBUG PRINT: Ghost Values (The Result) ---
    //printDebugExchangeResult();

}

// Perform Local SpMV (Compute y = Ax)
void Process::performLocalSpMV() {
    
    localY.assign(localCSR.nrows, 0.0);

    for (int i = 0; i < localCSR.nrows; i++) {
        double sum = 0.0;
        
        for (int k = localCSR.pointer[i]; k < localCSR.pointer[i+1]; k++) {
            int global_col = localCSR.index[k];
            double x_val;

            // CHECK: Is this column local or ghost?
            if (global_col % worldSize == rank) {
                // I own it! Math: local_index = global / worldSize
                x_val = localX[global_col / worldSize];
            } else {
                // It's a ghost! Use the map to find where we put it in the buffer
                // .at() is safer than [] because it throws an error if missing
                x_val = ghost_buffer[ghost_map.at(global_col)];
            }
            
            sum += localCSR.data[k] * x_val;
        }
        localY[i] = sum;
    }
}

// ----------------------------------------------------------------- //
// PRIVATE METHODS
void Process::calculateLocalRowsCols(CommMetadata &meta_x){
    for (int i = 0; i < total_rows; i++) { //total_rows = total_cols
        if (i % worldSize == rank) {
            localCSR.nrows++;
            meta_x.counts[rank]++;
        }
    }
}

void Process::calculateCountDispl(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x){
    // Calculate row length counts/displ (len_counts) for ALL ranks
    for(int i=0; i<worldSize; i++) {
        meta_len.counts[i] = (total_rows / worldSize) + (i < (total_rows % worldSize) ? 1 : 0);
        if(i > 0) meta_len.displs[i] = meta_len.displs[i-1] + meta_len.counts[i-1];
    }

    // ONLY ROOT needs to calculate send-counts for the Matrix and Vector X
    if(rank == ROOT_RANK){
        fill(meta_x.counts.begin(), meta_x.counts.end(), 0); // Reset
        // 1. Calculate Matrix NNZ counts
        for (int i = 0; i < total_rows; i++) {
            int target = i % worldSize;
            int nnzInRow = globalCSR.pointer[i+1] - globalCSR.pointer[i];
            meta_nnz.counts[target] += nnzInRow;
            meta_x.counts[target]++;
        }
        
        // Calculate displacements (offsets)
        for (int i = 1; i < worldSize; i++) {
            meta_nnz.displs[i] = meta_nnz.displs[i-1] + meta_nnz.counts[i-1];
            meta_x.displs[i] = meta_x.displs[i-1] + meta_x.counts[i-1];
        }
    }
    
}

void Process::shuffleVectors(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x, ShuffledBuffers &shuffled){

    // Track where we are writing for each rank within the shuffled buffers
    vector<int> current_pos = meta_nnz.displs; 
    vector<int> current_posX = meta_x.displs; 
    vector<int> write_ptr = meta_len.displs;

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

        shuffled.x[current_posX[target]] = globalX[i];
        current_posX[target]++;
    }
}

void Process::scattering(CommMetadata &meta_nnz, CommMetadata &meta_len, CommMetadata &meta_x, ShuffledBuffers &shuffled, vector<int> &local_lengths) {

    // Every rank needs to know its own NNZ count to resize its local arrays
    MPI_Scatter(meta_nnz.counts.data(), 1, MPI_INT, &localCSR.nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    int my_x_count = meta_x.counts[rank];

    localCSR.index.resize(localCSR.nnz);
    localCSR.data.resize(localCSR.nnz);
    local_lengths.resize(localCSR.nrows);
    localX.resize(my_x_count);

    MPI_Scatterv(shuffled.indices.data(), meta_nnz.counts.data(), meta_nnz.displs.data(), MPI_INT,
                localCSR.index.data(), localCSR.nnz, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.data.data(), meta_nnz.counts.data(), meta_nnz.displs.data(), MPI_DOUBLE,
             localCSR.data.data(), localCSR.nnz, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.lengths.data(), meta_len.counts.data(), meta_len.displs.data(), MPI_INT,
             local_lengths.data(), localCSR.nrows, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.x.data(), meta_x.counts.data(), meta_x.displs.data(), MPI_DOUBLE,
                 localX.data(), my_x_count, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
}

void Process::buildCSRpointer(vector<int> &local_lengths){
    localCSR.pointer.assign(localCSR.nrows + 1, 0);
    for (int i = 0; i < localCSR.nrows; i++) {
        localCSR.pointer[i+1] = localCSR.pointer[i] + local_lengths[i];
    }
}

// Initialize multiply vector X with random floating values
void Process::initGlobalX(vector<double>& v, int cols){
    random_device rd;  //Obtain a random number from hardware to seed the generator
    mt19937 gen(rd()); //Initialize the Mersenne Twister engine with the seed
    uniform_real_distribution<double> dis(std::nextafter(0.0, 100.0), 100.0); //Define the range (nextafter ensures 0 is not included)

    for (int i = 0; i < cols; i++) {
        v[i] = dis(gen);
    }
}

// =========================================================================
// PHASE 1: Identify what data we are missing ("Ghost Entries")
// =========================================================================
void Process::identifyGhostEntries(vector<vector<int>>& requests, map<int, int>& ghost_map) {
    // requests[target_rank] will store the list of global indices we need from them
    requests.resize(worldSize);
    
    // Use a set to avoid asking for the same index twice
    set<int> unique_ghosts;

    for (int col : localCSR.index) {
        // If I don't own this column, I need to ask for it
        if (col % worldSize != rank) {
            unique_ghosts.insert(col);
        }
    }

    // Sort them to ensure deterministic order (good for debugging)
    // and map them to a specific slot in our future 'ghost_buffer'
    int buffer_pos = 0;
    for (int global_col : unique_ghosts) {
        int owner = global_col % worldSize; //which rank has this colum?
        requests[owner].push_back(global_col); //at rank n will be requested the col
        
        // Save where this specific global index will live in our receive buffer
        ghost_map[global_col] = buffer_pos; 
        buffer_pos++;
    }

    // --- DEBUG PRINT: Requests and Ghost Map ---
    //printDebugIdentify(requests);
    // -------------------------------------------
}

// =========================================================================
// PHASE 2: Exchange Metadata (Tell neighbors how much we need)
// =========================================================================
void Process::exchangeMetadata(const vector<vector<int>>& requests, 
                               vector<int>& num_indices_I_request, 
                               vector<int>& num_indices_others_need_from_me) {
    
    num_indices_I_request.resize(worldSize);
    num_indices_others_need_from_me.resize(worldSize);

    // Fill the send buffer: "I am going to ask Rank i for X indices"
    for (int i = 0; i < worldSize; i++) {
        num_indices_I_request[i] = requests[i].size();
    }

    // Exchange counts. 
    // After this, num_indices_others_need_from_me[i] tells us:
    // "Rank i is about to send me a list of Y indices it wants."
    MPI_Alltoall(num_indices_I_request.data(), 1, MPI_INT, 
                 num_indices_others_need_from_me.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // --- DEBUG PRINT: Metadata Exchange (The Handshake) ---
    //printDebugExchangeMeta(num_indices_I_request, num_indices_others_need_from_me);
    // -----------------------------------------------------

}

// =========================================================================
// PHASE 3: The Handshake (Exchange Indices, then Exchange Values)
// =========================================================================
void Process::exchangeGhostValues(const vector<vector<int>>& requests, 
                                  const vector<int>& num_indices_I_request, 
                                  const vector<int>& num_indices_others_need_from_me,
                                  vector<double>& ghost_buffer) {

    // --- STEP A: PREPARE DISPLACEMENTS FOR MPI ---
    vector<int> send_displs(worldSize, 0); // For outgoing requests
    vector<int> recv_displs(worldSize, 0); // For incoming requests
    
    int total_requests_outgoing = 0;
    int total_requests_incoming = 0;

    for (int i = 0; i < worldSize; i++) {
        send_displs[i] = (i == 0) ? 0 : send_displs[i-1] + num_indices_I_request[i-1];
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + num_indices_others_need_from_me[i-1];
        
        total_requests_outgoing += num_indices_I_request[i];
        total_requests_incoming += num_indices_others_need_from_me[i];
    }

    // Flatten my 2D requests into a 1D array for MPI
    vector<int> flat_requests_outgoing;
    flat_requests_outgoing.reserve(total_requests_outgoing);
    for (const auto& req_vec : requests) {
        flat_requests_outgoing.insert(flat_requests_outgoing.end(), req_vec.begin(), req_vec.end());
    }

    // Buffer to hold the indices OTHER people want from ME
    vector<int> indices_others_want(total_requests_incoming);


    // --- STEP B: EXCHANGE THE INDICES (Ask for what we want) ---
    // We send 'flat_requests_outgoing' -> Others receive into 'indices_others_want'
    MPI_Alltoallv(flat_requests_outgoing.data(), num_indices_I_request.data(), send_displs.data(), MPI_INT,
                  indices_others_want.data(), num_indices_others_need_from_me.data(), recv_displs.data(), MPI_INT, 
                  MPI_COMM_WORLD);


    // --- STEP C: FETCH THE VALUES FROM MY LOCAL STORAGE ---
    // Now I know exactly which Global Indices others want. I must look them up.
    vector<double> values_others_need(total_requests_incoming);

    for (int i = 0; i < total_requests_incoming; i++) {
        int global_idx_requested = indices_others_want[i];
        
        // Convert Global Index -> Local Index (Cyclic Partitioning Logic)
        // If I own Global 4, 8, 12... and worldSize is 4:
        // Global 4 is at local index 1 (4 / 4 = 1).
        int local_idx = global_idx_requested / worldSize; 

        // Safety check (optional but recommended)
        if (local_idx < 0 || local_idx >= localX.size()) {
            printf("Error on Rank %d: Requested global %d maps to invalid local %d\n", rank, global_idx_requested, local_idx);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        values_others_need[i] = localX[local_idx];
    }

    // --- STEP D: SEND THE VALUES BACK (Reply to requests) ---
    ghost_buffer.resize(total_requests_outgoing);

    // Note carefully: The send/recv counts are FLIPPED here compared to Step B.
    // We are sending data back to satisfy the requests we just received.
    MPI_Alltoallv(values_others_need.data(), num_indices_others_need_from_me.data(), recv_displs.data(), MPI_DOUBLE,
                  ghost_buffer.data(), num_indices_I_request.data(), send_displs.data(), MPI_DOUBLE, 
                  MPI_COMM_WORLD);
}

// Debug methods

void Process::print() {
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 1. Global Summary (Rank 0 only) ---
    if (rank == ROOT_RANK) {
        printf("\n================ GLOBAL SUMMARY ================\n");
        printf("Matrix Dimensions : %d x %d\n", total_rows, total_rows);
        printf("Total Non-Zeros   : %d\n", total_nnz);
        
        // Print a preview of Global X (First 10 items)
        printf("Global X (Preview): [ ");
        int limit = (globalX.size() > 10) ? 10 : globalX.size();
        for (int i = 0; i < limit; i++) printf("%.2f ", globalX[i]);
        if (globalX.size() > 10) printf("... (%lu total) ", globalX.size());
        printf("]\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 2. Local Summaries (Rank by Rank) ---
    for (int r = 0; r < worldSize; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            printf("\n--- Rank %d | Rows: %d | NNZ: %d ---\n", 
                   rank, localCSR.nrows, (int)localCSR.index.size());

            // We skip printing local_pointer, local_index, and local_val 
            // because they are too huge. Instead, we show the RESULT (Y).

            // Preview Local Y (The Result)
            printf("  Computed Y (Preview): [ ");
            int limit = (localY.size() > 10) ? 10 : localY.size();
            for (int i = 0; i < limit; i++) {
                printf("%.2f ", localY[i]);
            }
            if (localY.size() > 10) printf("... (%lu total)", localY.size());
            printf("]\n");
            
            fflush(stdout); 
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == ROOT_RANK) cout << "\n================================================" << endl;
}

void Process::printDebugIdentify(vector<vector<int>> requests){
    if(rank==ROOT_RANK) cout << endl << "----- IDENTIFY PHASE -------" << endl;
    for (int r = 0; r < worldSize; r++) {
    MPI_Barrier(MPI_COMM_WORLD); // Wait for turn
    if (rank == r) {
        printf("\n=== Rank %d Identification Phase ===\n", rank);
        
        // 1. Print Requests (Who am I asking and for what?)
        printf("Requests to other ranks:\n");
        bool asking_anyone = false;
        for (int target = 0; target < worldSize; target++) {
            if (!requests[target].empty()) {
                asking_anyone = true;
                printf("  -> To Rank %d: [ ", target);
                // Print up to 10 indices to keep it readable
                int limit = (requests[target].size() > 10) ? 10 : requests[target].size();
                for (int i = 0; i < limit; i++) {
                    printf("%d ", requests[target][i]);
                }
                if (requests[target].size() > 10) printf("... (%lu total)", requests[target].size());
                printf("]\n");
            }
        }
        if (!asking_anyone) printf("  (None - I have all data locally)\n");

        // 2. Print Ghost Map (How do I find them later?)
        printf("Ghost Map (Global Index -> Buffer Position):\n");
        if (ghost_map.empty()) {
            printf("  (Empty)\n");
        } else {
            int count = 0;
            printf("  { ");
            for (auto const& [global_idx, buffer_pos] : ghost_map) {
                printf("%d:%d ", global_idx, buffer_pos);
                if (++count >= 10) { 
                    printf("... "); 
                    break; 
                }
            }
            printf("}\n");
        }
        fflush(stdout);
    }
    }
    MPI_Barrier(MPI_COMM_WORLD); // Final sync
}
void Process::printDebugExchangeMeta(const vector<int>& num_indices_I_request, const vector<int>& num_indices_others_need_from_me){
    if(rank==ROOT_RANK) cout << endl << "----- EXCHANGE METADATA PHASE -------" << endl;
    for (int r = 0; r < worldSize; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
        printf("\n=== Rank %d Metadata Phase ===\n", rank);
        
        // 1. What I am asking for (My Requests)
        printf("I am asking others for this many indices:\n");
        printf("  [ ");
        for (int count : num_indices_I_request) printf("%d ", count);
        printf("]\n");

        // 2. What others are asking me for (My Obligations)
        printf("Others are asking me for this many indices:\n");
        printf("  [ ");
        for (int count : num_indices_others_need_from_me) printf("%d ", count);
        printf("]\n");

        fflush(stdout);
    }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void Process::printDebugExchangeResult(){
    if(rank==ROOT_RANK) cout << endl << "----- EXCHANGE RESULT PHASE -------" << endl;
        for (int r = 0; r < worldSize; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            printf("\n=== Rank %d Ghost Exchange Results ===\n", rank);
            
            // 1. Print raw buffer contents
            printf("Raw Ghost Buffer (Size %lu):\n  [ ", ghost_buffer.size());
            for (double val : ghost_buffer) {
                printf("%.2f ", val);
            }
            printf("]\n");

            // 2. Print interpreted values (Global Index -> Value)
            printf("Decoded Values (Using Ghost Map):\n");
            if (ghost_map.empty()) {
                printf("  (No ghost entries needed)\n");
            } else {
                // ghost_map stores: Global_Index -> Position_in_Buffer
                for (auto const& [global_idx, buffer_pos] : ghost_map) {
                    // Safety check
                    if (buffer_pos < ghost_buffer.size()) {
                        double val = ghost_buffer[buffer_pos];
                        printf("  Global Index %d = %.2f (stored at buffer[%d])\n", 
                            global_idx, val, buffer_pos);
                    } else {
                        printf("  ERROR: Global Index %d maps to invalid buffer position %d\n", 
                            global_idx, buffer_pos);
                    }
                }
            }
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
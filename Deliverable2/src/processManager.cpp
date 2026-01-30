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

// =========================================================================
// Constructor and destructor
// =========================================================================
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

int Process::getRank(){
    return rank;
}
double Process::getCommTime(){
    return comm_time;
}
double Process::getCompTime(){
    return comp_time;
}

// =========================================================================
// PUBLIC METHODS
// =========================================================================
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

void Process::exchangeGhostIdentifier(){
    // Phase 1: Identify what data we are missing ("Ghost Entries")
    identifyGhostEntries(this->requests, this->ghost_map);
    
    // Phase 2: Handshake: exchange Metadata (Tell neighbors how much we need)
    exchangeMetadata(this->requests, this->send_counts, this->recv_counts);
}

void Process::prepareOptimizedStructures() {
    if(rank == ROOT_RANK) printf("Optimization: Renumbering columns and pre-calculating comms...\n");

    // --- STEP 1: PREPARE COMBINED X ---
    combinedX.resize(localX.size() + ghost_map.size());

    // --- STEP 2: RENUMBER COLUMNS ---
    renumbered_indices.resize(localCSR.index.size());
    for (size_t k = 0; k < localCSR.index.size(); k++) {
        int global_col = localCSR.index[k];
        if (global_col % worldSize == rank) {
            renumbered_indices[k] = global_col / worldSize;
        } else {
            renumbered_indices[k] = localX.size() + ghost_map[global_col];
        }
    }

    // --- STEP 3: PRE-CALCULATE COMMUNICATION ---
    
    // A. Prepare displacements
    // We need displacements for:
    // 1. My Requests (Ghosts) -> stored in 'send_counts'
    // 2. Others' Requests (Data) -> stored in 'recv_counts'
    
    // send_displs -> For 'send_counts' (Ghosts I Need)
    // recv_displs -> For 'recv_counts' (Data I Provide)
    
    send_displs.assign(worldSize, 0);
    recv_displs.assign(worldSize, 0);
    int total_ghosts_needed = 0;
    int total_data_provided = 0;

    for (int i = 0; i < worldSize; i++) {
        send_displs[i] = (i == 0) ? 0 : send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + recv_counts[i-1];
        
        total_ghosts_needed += send_counts[i]; 
        total_data_provided += recv_counts[i]; 
    }

    // B. Send the INDICES (The "Handshake")
    // sending: flat_requests (My Ghosts) -> Count: send_counts
    // recving: indices_others_want (Data others need) -> Count: recv_counts
    
    vector<int> flat_requests;
    for (const auto& req : requests) {
        flat_requests.insert(flat_requests.end(), req.begin(), req.end());
    }

    vector<int> indices_others_want_global(total_data_provided);

    MPI_Alltoallv(flat_requests.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  indices_others_want_global.data(), recv_counts.data(), recv_displs.data(), MPI_INT, 
                  MPI_COMM_WORLD);

    // C. Convert Global Indices to LOCAL Indices
    indices_to_send.resize(total_data_provided);
    for (int i = 0; i < total_data_provided; i++) {
        int g_idx = indices_others_want_global[i];
        indices_to_send[i] = g_idx / worldSize; 
    }

    ghost_buffer.resize(total_ghosts_needed);
}

// =========================================================================
// TRASNFER OF VALUES OF X ( Exchange Values)
// =========================================================================
void Process::exchangeGhostValues() {
    
    // 1. Pack the Buffer
    vector<double> send_buffer(indices_to_send.size());
    for (size_t i = 0; i < indices_to_send.size(); i++) {
        send_buffer[i] = localX[indices_to_send[i]];
    }

    // 2. Exchange Values
    // SENDING: send_buffer (Data) -> Use 'recv_counts' (Data I Provide) and 'recv_displs'
    // RECEIVING: ghost_buffer (Ghosts) -> Use 'send_counts' (Ghosts I Need) and 'send_displs'
    
    // Note: sdispls uses recv_displs because recv_displs tracks the layout of Data I Provide.
    // This naming is tricky but correct based on how we built the displacement vectors above.
    
    MPI_Alltoallv(send_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                  ghost_buffer.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, 
                  MPI_COMM_WORLD);
}

// Perform Local SpMV (Compute y = Ax)
void Process::runCalculation(int num_iter) {
    
    vector<double> total_comm(num_iter);
    vector<double> total_comp(num_iter);

    localY.assign(localCSR.nrows, 0.0);

    // --- THE MAIN SOLVER LOOP ---
    for (int iter = 0; iter < num_iter; iter++) {
        
        // A. MEASURE COMMUNICATION (The Ghost Exchange)
        MPI_Barrier(MPI_COMM_WORLD); // Optional: Sync for pure comm measurement
        
        double start = MPI_Wtime();
        exchangeGhostValues(); // Only exchanges the doubles!
        double end = MPI_Wtime();
        
        total_comm[iter] = (end - start)*1000; //ms

        // B. MEASURE COMPUTATION (The Math)
        // (No barrier needed here, we want to see if imbalance slows us down)
        start = MPI_Wtime();
        computelocalSpMV();
        end = MPI_Wtime();
        
        total_comp[iter] = (end - start)*1000; //ms
    }

    calculateP90(total_comm, total_comp);

}
// =========================================================================
// PRIVATE METHODS
// =========================================================================
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
void Process::identifyGhostEntries(vector<vector<int>>& requests, unordered_map<int, int>& ghost_map) {
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


void Process::computelocalSpMV() {
    
    // 1. Refresh the Unified Vector
    // Copy my fresh Local X into the start
    std::copy(localX.begin(), localX.end(), combinedX.begin());
    
    // Copy the fresh Ghost Values into the end
    std::copy(ghost_buffer.begin(), ghost_buffer.end(), combinedX.begin() + localX.size());

    // 2. The Loop 
    for (int i = 0; i < localCSR.nrows; i++) {
        double sum = 0.0;
        for (int k = localCSR.pointer[i]; k < localCSR.pointer[i+1]; k++) {
            // Use the renumbered index to jump directly to the value
            sum += localCSR.data[k] * combinedX[renumbered_indices[k]];
        }
        localY[i] = sum;
    }
}

//calculate P90 of times
void Process::calculateP90(vector<double> &local_comm, vector<double> &local_comp){
    
    int num_iter = local_comm.size();
    
    // 1. Gather the WORST time across all ranks for each iteration
    // We cannot just use Rank 0's time; we need the bottleneck time.
    vector<double> max_comm(num_iter);
    vector<double> max_comp(num_iter);

    MPI_Reduce(local_comm.data(), max_comm.data(), num_iter, MPI_DOUBLE, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(local_comp.data(), max_comp.data(), num_iter, MPI_DOUBLE, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);

    // 2. ONLY ROOT Calculates P90
    if (rank == ROOT_RANK) {
        
        // Sort to find percentiles
        std::sort(max_comm.begin(), max_comm.end());
        std::sort(max_comp.begin(), max_comp.end());

        // Calculate 90th percentile index (P90)
        // For 10 iterations, index 9. (0.9 * 10 = 9)
        int p90_index = (int)(0.9 * num_iter);
        if (p90_index >= num_iter) p90_index = num_iter - 1; // Safety clamp

        this->comm_time = max_comm[p90_index];
        this->comp_time = max_comp[p90_index];
        
        // Printing is handled by your main() later, or you can print here
    }
}

// =========================================================================
// DEBUG METHODS
// =========================================================================
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

        //globalCSR.print(); //print first 10 entries of globalCSR
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

void Process::printMetrics(){
    // --- METRIC 1: COMPUTATION (Load Balance) ---
    // The "work" is proportional to the Number of Non-Zeros (NNZ)
    long local_nnz = localCSR.index.size(); 
    
    long min_nnz, max_nnz, sum_nnz;

    // Gather stats to Root
    MPI_Reduce(&local_nnz, &min_nnz, 1, MPI_LONG, MPI_MIN, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &max_nnz, 1, MPI_LONG, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &sum_nnz, 1, MPI_LONG, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

    // --- METRIC 2: COMMUNICATION (Volume) ---
    // Volume = (Total doubles Sent) + (Total doubles Received)
    long local_comm_volume = 0;
    
    // Sum up everything in your send/recv lists
    for (int c : send_counts) local_comm_volume += c;
    for (int c : recv_counts) local_comm_volume += c;

    long min_comm, max_comm, sum_comm;

    MPI_Reduce(&local_comm_volume, &min_comm, 1, MPI_LONG, MPI_MIN, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_volume, &max_comm, 1, MPI_LONG, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_volume, &sum_comm, 1, MPI_LONG, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

    // --- PRINT RESULTS (Rank 0 only) ---
    if (rank == ROOT_RANK) {
        double avg_nnz = (double)sum_nnz / worldSize;
        double avg_comm = (double)sum_comm / worldSize;
        
        // Calculate Imbalance Ratio (1.0 is perfect, 2.0 means worst rank has 2x work of avg)
        double imbalance_nnz = (avg_nnz > 0) ? (double)max_nnz / avg_nnz : 0.0;

        // 1. Human Readable (Keep this for debugging/logs)
        printf("\n================ BONUS METRICS ================\n");
        printf("1. Computation Load (NNZ per Rank):\n");
        printf("   Min: %ld\n", min_nnz);
        printf("   Max: %ld\n", max_nnz);
        printf("   Avg: %.1f\n", avg_nnz);
        printf("   Imbalance Ratio: %.2f (Ideal = 1.0)\n", imbalance_nnz);
        
        printf("\n2. Communication Volume (Doubles Exchanged):\n");
        printf("   Min: %ld\n", min_comm);
        printf("   Max: %ld\n", max_comm);
        printf("   Avg: %.1f\n", avg_comm);
        printf("===============================================\n");

        // 2. MACHINE READABLE (Add this line for the bash script!)
        // Format: TAG min_nnz max_nnz avg_nnz imbalance_nnz min_comm max_comm avg_comm
        printf("BONUS_DATA %ld %ld %.2f %.2f %ld %ld %.2f\n", 
               min_nnz, max_nnz, avg_nnz, imbalance_nnz, 
               min_comm, max_comm, avg_comm);
    }
}




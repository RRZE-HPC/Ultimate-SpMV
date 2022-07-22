#ifndef STRUCTS
#define STRUCTS

#include "vectors.h"
#include <ctime>

template <typename VT, typename IT>
using V = Vector<VT, IT>;

void log(const char *log_msg, const clock_t begin_time = 0, const clock_t end_time = 0)
{
        std::fstream log_file_to_append;
        const std::string log_file_name = "log.txt";
        log_file_to_append.open(log_file_name, std::fstream::in | std::fstream::out | std::fstream::app);

        // Print header
        log_file_to_append << "[" << (long int)std::clock() << "]" << std::endl;

        // Print log message
        log_file_to_append << log_msg;

        // If timing measurement provided, print that as well
        if(end_time > 0 && begin_time > 0){
            log_file_to_append << ": " << double( end_time - begin_time ) / CLOCKS_PER_SEC << std::endl;
        }

        // Close the log
        log_file_to_append << "\n\n";
        log_file_to_append.close();
}

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

template <typename IT>
struct ContextData
{
    std::vector<IT> to_send_heri;
    std::vector<IT> local_needed_heri;

    std::vector<IT> shift_arr; //how does this work, with these on the heap?
    std::vector<IT> incidence_arr;

    // rank
    // comm_size
    // padding
    // what else?
};

template <typename VT, typename IT>
struct ScsData
{
    ST C{};
    ST sigma{};

    ST n_rows{};
    ST n_cols{};
    ST n_rows_padded{};
    ST n_chunks{};
    ST n_elements{}; // No. of nz + padding.
    ST nnz{};        // No. of nz only.

    V<IT, IT> chunk_ptrs;    // Chunk start offsets into col_idxs & values.
    V<IT, IT> chunk_lengths; // Length of one row in a chunk.
    V<IT, IT> col_idxs;
    V<VT, IT> values;
    V<IT, IT> old_to_new_idx;
};

struct Config
{
    long n_els_per_row{-1}; // ell
    long chunk_size{8};    // sell-c-sigma
    long sigma{1};         // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x{true};

    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A{false};

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{5};

    // Verify result of SpVM.
    bool validate_result{true};

    // Verify result against solution of COO kernel.
    bool verify_result_with_coo{true};

    // Print incorrect elements from solution.
    // bool verbose_verification{true};

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    bool sort_matrix{true};

    bool verbose_validation{false};

    bool log_prof{false};

    // Configures if the code will be executed in bench mode (b) or solve mode (s)
    char mode = 'b'; 

    // Selects the default matrix storage format
    std::string kernel_format = "scs"; 
};

template <typename VT, typename IT>
struct DefaultValues
{

    VT A{2.0};
    VT x{1.00};
    VT y{};

    VT *x_values{};
    ST n_x_values{};

    VT *y_values{};
    ST n_y_values{};
};

template <typename VT, typename IT>
struct BenchmarkResult
{
    double perf_mflops{};
    double mem_mb{};
    std::vector<double> perfs_from_procs; // used in Gather

    unsigned int size_value_type{};
    unsigned int size_index_type{};

    unsigned long n_calls{};
    double duration_total_s{};
    double duration_kernel_s{};

    bool is_result_valid{false};
    std::string notes;

    std::string value_type_str;
    std::string index_type_str;

    uint64_t value_type_size{};
    uint64_t index_type_size{};

    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    double fill_in_percent{};
    long C{};
    long sigma{};
    long nzr{};

    bool was_matrix_sorted{false};

    double mem_m_mb{};
    double mem_x_mb{};
    double mem_y_mb{};

    double beta{};

    double cb_a_0{};
    double cb_a_nzc{};

    std::vector<VT> y_out;
    std::vector<VT> x_out;
    std::vector<VT> total_spmvm_result;
    std::vector<VT> total_x;
};

// Honestly, probably not necessary
template <typename ST>
struct MtxDataBookkeeping
{
    ST n_rows{};
    ST n_cols{};
    ST nnz{};

    bool is_sorted{};
    bool is_symmetric{};
};

#endif
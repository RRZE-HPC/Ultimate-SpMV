#include "spmv.h"
#include "mtx-reader.h"
#include "vectors.h"
#include "splitSendMtxData.h"

#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

// The default C to use for sell-c-sigma, when no C is specified.
enum
{
    SCS_DEFAULT_C = 8
};

// Initialize all matrices and vectors the same.
// Use -rand to initialize randomly.
static bool g_same_seed_for_every_vector = true;

// Log information.
static bool g_log = false;

template <typename VT, typename IT>
using V = Vector<VT, IT>;
template <typename VT, typename IT>
using VG = VectorGpu<VT, IT>;

template <typename T>
struct max_rel_error
{
};

template <>
struct max_rel_error<float>
{
    using base_value_type = float;
    constexpr static float value = 1e-5f;
};
template <>
struct max_rel_error<double>
{
    using base_value_type = double;
    constexpr static double value = 1e-13;
};
template <>
struct max_rel_error<std::complex<float>>
{
    using base_value_type = float;
    constexpr static float value = 1e-5f;
};
template <>
struct max_rel_error<std::complex<double>>
{
    using base_value_type = double;
    constexpr static double value = 1e-13;
};

struct Config
{
    long n_els_per_row{-1}; // ell
    long chunk_size{SCS_DEFAULT_C};    // sell-c-sigma
    long sigma{1};         // sell-c-sigma

    // Initialize rhs vector with random numbers.
    bool random_init_x{true};
    // Override values of the matrix, read via mtx file, with random numbers.
    bool random_init_A{false};

    // No. of repetitions to perform. 0 for automatic detection.
    unsigned long n_repetitions{10};

    // Verify result of SpVM.
    bool verify_result{true};

    // Verify result against solution of COO kernel.
    bool verify_result_with_coo{true};

    // Print incorrect elements from solution.
    bool verbose_verification{true};

    // Sort rows/columns of sparse matrix before
    // converting it to a specific format.
    bool sort_matrix{true};

    // Configures if the code will be executed in bench mode or compute mode
    std::string mode = "bench"; 
};

template <typename VT, typename IT>
struct DefaultValues
{
    VT A{2.0};
    VT x{1.01};
    VT y{};

    VT *x_values{};
    ST n_x_values{};

    VT *y_values{};
    ST n_y_values{};
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

struct BenchmarkResult
{
    double perf_gflops{};
    double mem_mb{};

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
};

void log(const char *format, ...)
{
    if (g_log)
    {
        static double log_started = get_time();

        va_list args;
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "# [%10.4f] %s", get_time() - log_started, format);

        va_start(args, format);
        vprintf(buffer, args);
        va_end(args);

        fflush(stdout);
    }
}

template <ST C, typename VT, typename IT>
static void
scs_impl(const ST n_chunks,
         const IT * RESTRICT chunk_ptrs,
         const IT * RESTRICT chunk_lengths,
         const IT * RESTRICT col_idxs,
         const VT * RESTRICT values,
         const VT * RESTRICT x,
         VT * RESTRICT y)
{

    #pragma omp parallel for schedule(static)
    for (ST c = 0; c < n_chunks; ++c) {
        VT tmp[C]{};

        IT cs = chunk_ptrs[c];

        for (IT j = 0; j < chunk_lengths[c]; ++j) {
            #pragma omp simd
            for (IT i = 0; i < C; ++i) {
                tmp[i] += values[cs + j * C + i] * x[col_idxs[cs + j * C + i]];
            }
        }

        #pragma omp simd
        for (IT i = 0; i < C; ++i) {
            y[c * C + i] = tmp[i];
        }
    }
}


/**
 * Dispatch to Sell-C-sigma kernels templated by C.
 *
 * Note: only works for selected Cs, see INSTANTIATE_CS.
 */
template <typename VT, typename IT>
static void
spmv_omp_scs_c(
             const ST C,
             const ST n_chunks,
             const IT * RESTRICT chunk_ptrs,
             const IT * RESTRICT chunk_lengths,
             const IT * RESTRICT col_idxs,
             const VT * RESTRICT values,
             const VT * RESTRICT x,
             VT * RESTRICT y)
{
    switch (C)
    {
        #define INSTANTIATE_CS X(2) X(4) X(8) X(16) X(32) X(64)

        #define X(CC) case CC: scs_impl<CC>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y); break;
        INSTANTIATE_CS
        #undef X

#ifdef SCS_C
    case SCS_C:
        case SCS_C: scs_impl<SCS_C>(n_chunks, chunk_ptrs, chunk_lengths, col_idxs, values, x, y);
        break;
#endif
    default:
        fprintf(stderr,
                "ERROR: for C=%ld no instantiation of a sell-c-sigma kernel exists.\n",
                long(C));
        exit(1);
    }
}

template <typename VT>
static void
print_vector(const std::string &name,
             const VT *begin,
             const VT *end)
{
    std::cout << name << " [" << end - begin << "]:";
    for (const VT *it = begin; it != end; ++it)
    {
        std::cout << " " << *it;
    }
    std::cout << "\n";
}

template <typename VT, typename IT>
static void
print_vector(const std::string &name,
             const V<VT, IT> &v)
{
    print_vector(name, v.data(), v.data() + v.n_rows);
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
static bool
will_add_overflow(T a, T b)
{
    if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() - a < b;
    }
    else if (a < 0 && b < 0)
    {
        return std::numeric_limits<T>::min() - a > b;
    }

    return false;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_unsigned<T>::value,
              bool>::type = true>
static bool
will_add_overflow(T a, T b)
{
    return std::numeric_limits<T>::max() - a < b;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
static bool
will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }
    else if (a < 0 && b > 0)
    {
        return std::numeric_limits<T>::min() / b > a;
    }
    else if (a > 0 && b < 0)
    {
        return std::numeric_limits<T>::min() / a > b;
    }
    else if (a > 0 && b > 0)
    {
        return std::numeric_limits<T>::max() / a < b;
    }
    else
    {
        T difference =
            std::numeric_limits<T>::max() + std::numeric_limits<T>::min();

        if (difference == 0)
        { // symmetric case
            return std::numeric_limits<T>::min() / a < b * T{-1};
        }
        else
        { // abs(min) > max
            T c = std::numeric_limits<T>::min() - difference;

            if (a < c || b < c)
                return true;

            T ap = a * T{-1};
            T bp = b * T{-1};

            return std::numeric_limits<T>::max() / ap < bp;
        }
    }
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_unsigned<T>::value,
              bool>::type = true>
static bool
will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }

    return std::numeric_limits<T>::max() / a < b;
}

static std::tuple<std::string, uint64_t>
type_info_from_type_index(const std::type_index &ti)
{
    static std::unordered_map<std::type_index, std::tuple<std::string, uint64_t>> type_map = {
        {std::type_index(typeid(double)), std::make_tuple("dp", sizeof(double))},
        {std::type_index(typeid(float)), std::make_tuple("sp", sizeof(float))},

        {std::type_index(typeid(int)), std::make_tuple("int", sizeof(int))},
        {std::type_index(typeid(long)), std::make_tuple("long", sizeof(long))},
        {std::type_index(typeid(int32_t)), std::make_tuple("int32_t", sizeof(int32_t))},
        {std::type_index(typeid(int64_t)), std::make_tuple("int64_t", sizeof(int64_t))},

        {std::type_index(typeid(unsigned int)), std::make_tuple("uint", sizeof(unsigned int))},
        {std::type_index(typeid(unsigned long)), std::make_tuple("ulong", sizeof(unsigned long))},
        {std::type_index(typeid(uint32_t)), std::make_tuple("uint32_t", sizeof(uint32_t))},
        {std::type_index(typeid(uint64_t)), std::make_tuple("uint64_t", sizeof(uint64_t))}};

    auto it = type_map.find(ti);

    if (it == type_map.end())
    {
        return std::make_tuple(std::string{"unknown"}, uint64_t{0});
    }

    return it->second;
}

static std::string
type_name_from_type_index(const std::type_index &ti)
{
    return std::get<0>(type_info_from_type_index(ti));
}

#if 0
// Currently unused
static uint64_t
type_size_from_type_index(const std::type_index & ti)
{
    return std::get<1>(type_info_from_type_index(ti));
}
#endif

template <typename T>
static std::string
type_name_from_type()
{
    return type_name_from_type_index(std::type_index(typeid(T)));
}

class Histogram
{
public:
    Histogram()
        : Histogram(29)
    {
    }

    Histogram(size_t n_buckets)
    {
        bucket_upper_bounds().push_back(0);

        int start = 1;
        int end = start * 10;
        int inc = start;

        while (n_buckets > 0)
        {

            for (int i = start; i < end && n_buckets > 0; i += inc, --n_buckets)
            {
                bucket_upper_bounds().push_back(i);
            }

            start = end;
            end = start * 10;
            inc = start;

            // if (n_buckets < 9) {
            //     n_buckets = 0;
            // }
            // else {
            //     n_buckets -= 9;
            // }
        }

        bucket_upper_bounds().push_back(end + 1);
        // for (int i = 1; i < 10; ++i) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // for (int i = 10; i < 100; i += 10) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // for (int i = 100; i < 1000; i += 100) {
        //     bucket_upper_bounds().push_back(i);
        // }
        // bucket_upper_bounds().push_back(1000000);
        bucket_counts().resize(bucket_upper_bounds().size());
    }

    // Option -Ofast optimizes too much here in some cases, so revert to
    // no optimization.
    // (pow gets replaced by exp( ln(10.0) * bp ) with ln(10.0) as a constant)

    void
        __attribute__((optimize("O0")))
        insert(int64_t value)
    {
        if (value < 0)
        {
            // if (value < 1) {
            throw std::invalid_argument("value must be > 0");
        }

        // remove if 0 should not be counted

        if (value == 0)
        {
            bucket_counts()[0] += 1;
            return;
        }
        else if (value == 1)
        {
            bucket_counts()[1] += 1;
            return;
        }

        value -= 1;

        double x = std::log10(static_cast<double>(value));
        double bp = std::floor(x);

        size_t inner_index = (size_t)std::floor(static_cast<double>(value) / std::pow(10.0, bp));
        size_t outer_index = 9 * (size_t)(bp);

        // decrement by 1 (-1) when value == 0 should not be counted!

        size_t index = outer_index + inner_index; // TODO: - 1ul;
        // shift index, as we manually insert 0.
        index += 1;

        if (index >= bucket_counts().size())
        {
            index = bucket_counts().size() - 1ul;
        }

        bucket_counts()[index] += 1;
    }

    std::vector<uint64_t> &bucket_counts() { return bucket_counts_; }
    std::vector<int64_t> &bucket_upper_bounds() { return bucket_upper_bounds_; }

    const std::vector<uint64_t> &bucket_counts() const { return bucket_counts_; }
    const std::vector<int64_t> &bucket_upper_bounds() const { return bucket_upper_bounds_; }

private:
    std::vector<uint64_t> bucket_counts_;
    std::vector<int64_t> bucket_upper_bounds_;
};

template <typename T = double>
struct Statistics
{
    T min{std::numeric_limits<T>::max()};
    T max{std::numeric_limits<T>::min()};
    T avg{};
    T std_dev{};
    T cv{};
    T median{};

    Histogram hist{};
};

template <typename T = double>
struct RowOrColStats
{
    T value_min{std::numeric_limits<T>::max()};
    T value_max{std::numeric_limits<T>::min()};
    int n_values{};
    int n_non_zeros{};
    uint64_t min_idx{std::numeric_limits<uint64_t>::max()};
    uint64_t max_idx{};
};

template <typename T = double>
struct MatrixStats
{
    std::vector<RowOrColStats<T>> all_rows;
    std::vector<RowOrColStats<T>> all_cols;

    Statistics<T> row_lengths{};
    Statistics<T> col_lengths{};
    T densitiy{};

    uint64_t n_rows{};
    uint64_t n_cols{};
    uint64_t nnz{};

    Statistics<T> bandwidths;

    bool is_symmetric{};
    bool is_sorted{};
};

template <typename T = double, typename U = double, typename F>
static struct Statistics<T>
get_statistics(
    std::vector<U> &entries,
    F &getter,
    size_t n_buckets_in_histogram = 29)
{
    Statistics<T> s{};
    s.hist = Histogram(n_buckets_in_histogram);
    T sum{};

    // min/max/avg

    for (const auto &entry : entries)
    {
        T n_values = getter(entry);

        sum += n_values;

        if (s.max < n_values)
        {
            s.max = n_values;
        }
        if (s.min > n_values)
        {
            s.min = n_values;
        }
    }

    s.avg = sum / (T)entries.size();

    // std deviation, cv

    T sum_squares{};

    for (const auto &entry : entries)
    {
        T n_values = (T)getter(entry);
        sum_squares += (n_values - s.avg) * (n_values - s.avg);
    }

    s.std_dev = std::sqrt(sum_squares / (T)entries.size());
    s.cv = s.std_dev / s.avg;

    // WARNING: entries will be sorted...
    auto median = [&getter](std::vector<U> &entries) -> int
    {
        auto middle_el = begin(entries) + entries.size() / 2;
        std::nth_element(begin(entries), middle_el, end(entries),
                         [&getter](const auto &a, const auto &b)
                         {
                             return getter(a) < getter(b);
                         });

        return getter(entries[entries.size() / 2]);
    };

    s.median = median(entries);

    // Fill histogram

    for (const auto &entry : entries)
    {
        s.hist.insert(getter(entry));
    }

    return s;
}

// TODO: Make work for multiple procs
template <typename VT, typename IT>
static MatrixStats<double>
get_matrix_stats(const MtxData<VT, IT> &mtx)
{
    // int my_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MatrixStats<double> stats;
    auto &all_rows = stats.all_rows;
    auto &all_cols = stats.all_cols;

    all_rows.resize(mtx.n_rows);
    all_cols.resize(mtx.n_cols); // forces all_cols container to have n_cols elements

    // std::vector<IT> local_J;
    // IT updated_col;
    // auto minimum_col = *std::min_element(mtx.J.begin(), mtx.J.end());

    // for(int i = 0; i < mtx.nnz; ++i){
    //     updated_col = mtx.J[i] - minimum_col - 1;

    //     local_J.push_back(updated_col);
    // }

    for (ST i = 0; i < mtx.nnz; ++i)
    {
        IT row = mtx.I[i];
        // IT col = local_J[i];  // for some reason, works when original method fails.
        IT col = mtx.J[i];

        if (mtx.values[i] != VT{})
        {
            // Skip if we do not want information about the spread of the values.
            if (all_rows[row].value_min > fabs(mtx.values[i]))
            {
                all_rows[row].value_min = fabs(mtx.values[i]);
            }

            if (all_rows[row].value_max < fabs(mtx.values[i]))
            {
                all_rows[row].value_max = fabs(mtx.values[i]);
            }

            if (all_cols[col].value_min > fabs(mtx.values[i]))
            {
                all_cols[col].value_min = fabs(mtx.values[i]);
            }

            if (all_cols[col].value_max < fabs(mtx.values[i]))
            {
                all_cols[col].value_max = fabs(mtx.values[i]);
            }
            // Skip end

            ++all_rows[row].n_non_zeros;
            ++all_cols[col].n_non_zeros;
        }

        ++all_rows[row].n_values;
        ++all_cols[col].n_values;

        // if ((uint64_t)col > all_rows[row].max_idx) {
        //     all_rows[row].max_idx = col;
        // }
        // if ((uint64_t)col < all_rows[row].min_idx) {
        //     all_rows[row].min_idx = col;
        // }

        // if ((uint64_t)row > all_rows[col].max_idx) {
        //     all_rows[col].max_idx = row;
        // }
        // if ((uint64_t)row < all_rows[col].min_idx) {
        //     all_rows[col].min_idx = row;
        // }

        // printf("(Proc: %i, nnz %li out of %li, row %i, col %i) => %li %li\n", my_rank, i, mtx.nnz, row, col, all_rows[col].min_idx, all_rows[col].max_idx);
    }
    // compute bandwidth and histogram for bandwidth from row stats
    {
        std::vector<uint64_t> bandwidths;
        bandwidths.reserve(mtx.n_rows);

        for (uint64_t row_idx = 0; row_idx < (uint64_t)stats.all_rows.size(); ++row_idx)
        {
            const auto &row = stats.all_rows[row_idx];

            uint64_t local_bw = 1;

            if (row_idx > row.min_idx)
                local_bw += row_idx - row.min_idx;
            if (row_idx < row.max_idx)
                local_bw += row.max_idx - row_idx;

            bandwidths.push_back(local_bw);
        }

        auto get_el = [](const uint64_t &e)
        { return (double)e; };

        // determine needed no. of buckets in histogram
        size_t n_buckets = std::ceil(std::log10(mtx.n_cols) * 9.0 + 1.0);
        stats.bandwidths = get_statistics<double>(bandwidths, get_el, n_buckets);
    }

    auto get_n_values = [](const RowOrColStats<double> &e)
    { return e.n_values; };

    stats.row_lengths = get_statistics<double>(stats.all_rows, get_n_values);
    stats.col_lengths = get_statistics<double>(stats.all_cols, get_n_values);

    stats.densitiy = (double)mtx.nnz / ((double)mtx.n_rows * (double)mtx.n_cols);
    stats.n_rows = mtx.n_rows;
    stats.n_cols = mtx.n_cols;
    stats.nnz = mtx.nnz;
    stats.is_symmetric = mtx.is_symmetric;
    stats.is_sorted = mtx.is_sorted;
    return stats;
}

template <typename IT>
static void
convert_idxs_to_ptrs(const std::vector<IT> &idxs,
                     V<IT, IT> &ptrs)
{
    std::fill(ptrs.data(), ptrs.data() + ptrs.n_rows, 0);

    for (const auto idx : idxs)
    {
        if (idx + 1 < ptrs.n_rows)
        {
            ++ptrs[idx + 1];
        }
    }

    std::partial_sum(ptrs.data(), ptrs.data() + ptrs.n_rows, ptrs.data());
}

/**
 * Compute maximum number of elements in a row.
 * \p num_rows: Number of rows.
 * \p nnz: Number of non-zeros, also number of elements in \p row_indices.
 * \p row_indices: Array with row indices.
 */
template <typename IT>
static IT
calculate_max_nnz_per_row(
    ST num_rows, ST nnz,
    const IT *row_indices)
{
    std::vector<IT> rptr(num_rows + 1);

    for (ST i = 0; i < nnz; ++i)
    {
        IT row = row_indices[i];
        if (row + 1 < num_rows)
        {
            ++rptr[row + 1];
        }
    }

    return *std::max_element(rptr.begin(), rptr.end());
}

/**
 * Convert \p mtx to sell-c-sigma data structures.
 *
 * If \p C is < 1 then SCS_DEFAULT_C is used.
 * If \p sigma is < 1 then 1 is used.
 *
 * Note: the matrix entries in \p mtx don't need to be sorted.
 */
template <typename VT, typename IT>
static bool
convert_to_scs(const MtxData<VT, IT> & mtx,
               ST C, ST sigma,
               ScsData<VT, IT> & d)
{
    d.nnz    = mtx.nnz;
    d.n_rows = mtx.n_rows;
    d.n_cols = mtx.n_cols;

    d.C = C;
    d.sigma = sigma;

    if (d.sigma % d.C != 0 && d.sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(d.n_rows, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_chunks      = (mtx.n_rows + d.C - 1) / d.C;

    if (will_mult_overflow(d.n_chunks, d.C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        return false;
    }
    d.n_rows_padded = d.n_chunks * d.C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(d.n_rows_padded);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < mtx.nnz; ++i) {
        ++n_els_per_row[mtx.I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(d.n_rows_padded, d.sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        return false;
    }

    for (ST i = 0; i < d.n_rows_padded; i += d.sigma) {
        auto begin = &n_els_per_row[i];
        auto end   = (i + d.sigma) < d.n_rows_padded
                        ? &n_els_per_row[i + d.sigma]
                        : &n_els_per_row[d.n_rows_padded];

        std::sort(begin, end,
                  // sort longer rows first
                  [](const auto & a, const auto & b) {
                    return a.second > b.second;
                  });
    }

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    d.chunk_lengths = V<IT, IT>(d.n_chunks); // init a vector of length d.n_chunks
    d.chunk_ptrs    = V<IT, IT>(d.n_chunks + 1);

    IT cur_chunk_ptr = 0;
    
    for (ST i = 0; i < d.n_chunks; ++i) {
        auto begin = &n_els_per_row[i * d.C];
        auto end   = &n_els_per_row[i * d.C + d.C];

        d.chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, d.chunk_lengths[i] * (IT)d.C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            return false;
        }

        d.chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += d.chunk_lengths[i] * d.C;
    }

    

    ST n_scs_elements = d.chunk_ptrs[d.n_chunks - 1]
                        + d.chunk_lengths[d.n_chunks - 1] * d.C;
    d.chunk_ptrs[d.n_chunks] = n_scs_elements;

    // construct permutation vector

    d.old_to_new_idx = V<IT, IT>(d.n_rows);

    for (ST i = 0; i < d.n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < d.n_rows) {
            d.old_to_new_idx[old_row_idx] = i;
        }
    }

    

    d.values   = V<VT, IT>(n_scs_elements);
    d.col_idxs = V<IT, IT>(n_scs_elements);

    for (ST i = 0; i < n_scs_elements; ++i) {
        d.values[i]   = VT{};
        d.col_idxs[i] = IT{};
    }

    std::vector<IT> col_idx_in_row(d.n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < d.nnz; ++i) {
        IT row_old = mtx.I[i];
        IT row = d.old_to_new_idx[row_old];

        ST chunk_index = row / d.C;

        IT chunk_start = d.chunk_ptrs[chunk_index];
        IT chunk_row   = row % d.C;

        IT idx = chunk_start + col_idx_in_row[row] * d.C + chunk_row;

        d.col_idxs[idx] = mtx.J[i];
        d.values[idx]   = mtx.values[i];

        col_idx_in_row[row]++;
    }

    d.n_elements = n_scs_elements;

    return true;
}

/**
 * Compare vectors \p reference and \p actual.  Return the no. of elements that
 * differ.
 */
template <typename VT>
static ST
compare_arrays(const VT *reference,
               const VT *actual, const ST n,
               const bool verbose,
               const VT max_rel_error,
               VT &max_rel_error_found)
{
    ST error_counter = 0;
    max_rel_error_found = VT{};

    for (ST i = 0; i < n; ++i)
    {
        VT rel_error = std::abs((actual[i] - reference[i]) / reference[i]);

        if (rel_error > max_rel_error)
        {
            if (verbose && error_counter < 10)
            {
                std::fprintf(stderr,
                             "  expected element %2ld = %19.12e, but got %19.13e, rel. err. %19.12e\n",
                             (long)i, reference[i], actual[i], rel_error);
            }
            ++error_counter;
        }

        if (max_rel_error_found < rel_error)
        {
            max_rel_error_found = rel_error;
        }
    }

    if (verbose && error_counter > 0)
    {
        std::fprintf(stderr, "  %ld/%ld elements do not match\n", (long)error_counter,
                     (long)n);
    }

    return error_counter;
}

// template <typename VT>
// static bool
// spmv_verify(
//     const VT *y_ref,
//     const VT *y_actual,
//     const ST n,
//     bool verbose)
// {
//     VT max_rel_error_found{};

//     ST error_counter =
//         compare_arrays(
//             y_ref, y_actual, n,
//             verbose,
//             max_rel_error<VT>::value,
//             max_rel_error_found);

//     if (error_counter > 0)
//     {
//         // TODO: fix reported name and sizes.
//         fprintf(stderr,
//                 "WARNING: spmv kernel %s (fp size %lu, idx size %lu) is incorrect, "
//                 "relative error > %e for %ld/%ld elements. Max found rel error %e.\n",
//                 "", sizeof(VT), 0ul,
//                 max_rel_error<VT>::value,
//                 (long)error_counter, (long)n,
//                 max_rel_error_found);
//     }

//     return error_counter == 0;
// }

// template <typename VT, typename IT>
// static bool
// spmv_verify(const std::string &kernel_name,
//             const MtxData<VT, IT> &mtx,
//             const std::vector<VT> &x,
//             const std::vector<VT> &y_actual)
// {
//     std::vector<VT> y_ref(mtx.n_rows);

//     ST nnz = mtx.nnz;
//     if (mtx.I.size() != mtx.J.size() || mtx.I.size() != mtx.values.size())
//     {
//         fprintf(stderr, "ERROR: %s:%d sizes of rows, cols, and values differ.\n", __FILE__, __LINE__);
//         exit(1);
//     }

//     for (ST i = 0; i < nnz; ++i)
//     {
//         y_ref[mtx.I[i]] += mtx.values[i] * x[mtx.J[i]];
//     }

//     return spmv_verify(y_ref.data(), y_actual.data(),
//                        y_actual.size(), /*verbose*/ true);
// }

// template <typename VT, typename IT, typename FN>
// static BenchmarkResult
// spmv(FN &&kernel, bool is_gpu_kernel, const Config &config)
// {
//     log("running kernel begin\n");

//     log("warmup begin\n"); // what is the purpose of this warm-up?

//     kernel();

//     if (is_gpu_kernel)
//     {
// #ifdef __NVCC__
//         assert_gpu(cudaDeviceSynchronize());
// #endif
//     }

//     log("warmup end\n");

//     double t_kernel_start = 0.0;
//     double t_kernel_end = 0.0;
//     double duration = 0.0;

//     // Indicate if result is invalid, e.g., duration of >1s was not reached.
//     bool is_result_valid = true;

//     unsigned long n_repetitions = config.n_repetitions > 0 ? config.n_repetitions : 5;
//     int repeate_measurement;

//     do
//     {
//         log("running kernel with %ld repetitions\n", n_repetitions);
//         repeate_measurement = 0;

//         t_kernel_start = get_time();

//         /* M AND P */
//         // If wa want to hard-code number of iterations, it would be done here
//         // n_repetitions = 10;
//         for (unsigned long r = 0; r < n_repetitions; ++r)
//         {
//             // i.e. do kernel() for every repetition
//             // std::cout << "Kernal run: " << r << std::endl;
//             kernel();
//             // this is where swapping goes!
//             // swap(x,y), but where are x and y?
//         }
//         if (is_gpu_kernel)
//         {
// #ifdef __NVCC__
//             assert_gpu(cudaDeviceSynchronize());
// #endif
//         }

//         t_kernel_end = get_time();

//         duration = t_kernel_end - t_kernel_start;

//         if (duration < 1.0 && config.n_repetitions == 0)
//         {
//             unsigned long prev_n_repetitions = n_repetitions;
//             n_repetitions = std::ceil(n_repetitions / duration * 1.1);

//             if (prev_n_repetitions == n_repetitions)
//             {
//                 ++n_repetitions;
//             }

//             if (n_repetitions < prev_n_repetitions)
//             {
//                 // This typically happens if type ulong is too small to hold the
//                 // number of repetitions we would need for a duration > 1s.
//                 // We use the time we measured and flag the result.
//                 log("cannot increase no. of repetitions any further to reach a duration of >1s\n");
//                 log("aborting measurement for this kernel\n");
//                 n_repetitions = prev_n_repetitions;
//                 repeate_measurement = 0;
//                 is_result_valid = false;
//             }
//             else
//             {
//                 repeate_measurement = 1;
//             }
//         }
//     } while (repeate_measurement);

//     BenchmarkResult r;

//     r.is_result_valid = is_result_valid;
//     r.n_calls = n_repetitions;
//     r.duration_total_s = duration;

//     log("running kernel end\n");

//     return r;
// }

template <typename T, typename DIST, typename ENGINE>
struct random_number
{
    static T
    get(DIST &dist, ENGINE &engine)
    {
        return dist(engine);
    }
};

template <typename DIST, typename ENGINE>
struct random_number<std::complex<float>, DIST, ENGINE>
{
    static std::complex<float>
    get(DIST &dist, ENGINE &engine)
    {
        return std::complex<float>(dist(engine), dist(engine));
    }
};

template <typename DIST, typename ENGINE>
struct random_number<std::complex<double>, DIST, ENGINE>
{
    static std::complex<double>
    get(DIST &dist, ENGINE &engine)
    {
        return std::complex<double>(dist(engine), dist(engine));
    }
};

template <typename VT>
static void
random_init(VT *begin, VT *end)
{
    std::mt19937 engine;

    if (!g_same_seed_for_every_vector)
    {
        std::random_device rnd_device;
        engine.seed(rnd_device());
    }

    std::uniform_real_distribution<double> dist(0.1, 2.0);

    for (VT *it = begin; it != end; ++it)
    {
        *it = random_number<VT, decltype(dist), decltype(engine)>::get(dist, engine);
    }
}

template <typename VT, typename IT>
static void
random_init(V<VT, IT> &v)
{
    random_init(v.data(), v.data() + v.n_rows);
}

template <typename VT, typename IT>
static void
init_with_ptr_or_value(V<VT, IT> &x,
                       ST n_x,
                       const std::vector<VT> *x_in,
                       VT default_value,
                       bool init_with_random_numbers = false)
{
    if (!init_with_random_numbers)
    {
        if (x_in)
        {
            if (x_in->size() != size_t(n_x))
            {
                fprintf(stderr, "ERROR: x_in has incorrect size.\n");
                exit(1);
            }

            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = (*x_in)[i];
            }
        }
        else
        {
            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = default_value;
            }
        }
    }
    else
    {
        random_init(x);
    }
}

template <typename VT>
static void
init_std_vec_with_ptr_or_value(std::vector<VT> &x,
                               ST n_x,
                               const std::vector<VT> *x_in,
                               VT default_value,
                               bool init_with_random_numbers = false)
{
    if (!init_with_random_numbers)
    {
        if (x_in)
        {
            if (x_in->size() != size_t(n_x))
            {
                fprintf(stderr, "ERROR: x_in has incorrect size.\n");
                exit(1);
            }

            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = (*x_in)[i];
            }
        }
        else
        {
            for (ST i = 0; i < n_x; ++i)
            {
                x[i] = default_value;
            }
        }
    }
    else
    {
        random_init(&(*x.begin()), &(*x.end()));
    }
}

// TODO: remove VT, only needed for debugging?
template <typename VT, typename IT>
void collect_local_needed_heri(std::vector<IT> *local_needed_heri, const MtxData<VT, IT> &local_mtx, const int *work_sharing_arr){
    /* Here, we collect the row indicies of the halo elements needed for this process to have a valid local_x_scs to perform the SPMVM.
    These are organized as tuples in a vector, of the form (proc to, proc from, global row idx). The row_idx
    refers to the "global" x vector, this will be adjusted (localized) later when said element is "retrieved".
    The "proc" of where this needed element resides is known from the work sharing array.
    Each process needs to know the required halo elements for every other process. */

    // TODO: better to pass as args to function?
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // if(my_rank == 0){
    //     for(int j = 0; j < comm_size + 1; ++j)
    //         printf("%i\n", work_sharing_arr[j]);
    // //         std::cout << values.size() << std::endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    //nnz and values.size() should be the same
    // printf("%i ==? %i\n", *nnz, *values);

    IT from_proc, total_x_row_idx, remote_elem_candidate_col, remote_elem_col;
    // std::vector<std::tuple<IT, IT, IT>> global_needed_heri;
    int needed_heri_count = 0;
    // std::tuple<IT, IT, IT> needed_tup;

    // First, assemble the global view of required elements
    // for(int rank = 0; rank < comm_size; ++rank){ // TODO: Would love to parallelize
    for(int i = 0; i < local_mtx.nnz; ++i){ // this is only MY nnz, not the nnz of the process_local_mtx were looking at
        // If true, this is a remote element, and needs to be added to vector
        if(local_mtx.J[i] < work_sharing_arr[my_rank] || local_mtx.J[i] > work_sharing_arr[my_rank + 1]){
            // printf("Remote Element!\n");
            remote_elem_col = local_mtx.J[i];
            // Deduce from which process the required remote element lies
            for(int j = 0; j < comm_size; ++j){
                // printf("%i, %i, %i\n", remote_elem_col, work_sharing_arr[j], work_sharing_arr[j + 1]);
                // NOTE: not comepletly sure about about cases on the edge here
                if(remote_elem_col >= work_sharing_arr[j] && remote_elem_col < work_sharing_arr[j + 1]){
                    // printf("I should be here twice\n");
                    from_proc = j;
                    local_needed_heri->push_back(my_rank);
                    local_needed_heri->push_back(from_proc);
                    local_needed_heri->push_back(remote_elem_col);
                    ++needed_heri_count;
                }
            }
        }
        // std::cout << "remote element at loc " << local_mtx.values[i] << " and proc " << my_rank << std::endl;
        // std::cout << "column idx can only be between " << work_sharing_arr[my_rank] << " and " << work_sharing_arr[my_rank + 1] << std::endl;
        // std::cout << "col idx: " << local_mtx.J[i] << std::endl;
    }
// }


    // Once each process knows what it needs to recieve, and what it needs to send, the MPI comm should be easy 
}

// TODO: remove VT, only needed for debugging?
template <typename VT, typename IT>
void collect_to_send_heri(std::vector<IT> *to_send_heri, std::vector<IT> *local_needed_heri, const MtxData<VT, IT> &local_mtx, const int *work_sharing_arr){

    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Make every process aware of the size of the global_needed_heri array
    int local_needed_heri_size = local_needed_heri->size();
    int global_needed_heri_size = 0;

    // std::cout << local_needed_heri_size << std::endl;
    // exit(0);

    // First, gather the sizes of messages for the Allgetherv
    // NOTE: This assumes that the order of ranks is maintained
    int *all_local_needed_heri_sizes = new int[comm_size];

    MPI_Allgather(&local_needed_heri_size,
                1,
                MPI_INT,
                all_local_needed_heri_sizes,
                1,
                MPI_INT,
                MPI_COMM_WORLD);

    // Second, sum this array, which will be the size of the global_needed_heri_size
    int *global_needed_heri_displ_arr = new int[comm_size];
    for(int i = 0; i < comm_size; ++i){
        global_needed_heri_displ_arr[i] = global_needed_heri_size;
        // global_needed_heri_displ_arr[i] = 0;

        // if(my_rank == 2){std::cout << global_needed_heri_size << std::endl;}
        global_needed_heri_size += all_local_needed_heri_sizes[i];
    }
    global_needed_heri_displ_arr[comm_size] = global_needed_heri_size; //I need this?
    // if(my_rank == 2){std::cout << global_needed_heri_size << std::endl;}
    // Alternatively, Allreduce the local sizes
    // MPI_Allreduce(&local_needed_heri_size,
    //               &global_needed_heri_size,
    //               1,
    //               MPI_INT,
    //               MPI_SUM,
    //               MPI_COMM_WORLD);

    // TODO: what are the causes and implications of global_needed_heri_size?
    int *global_needed_heri = new int[global_needed_heri_size];

    // if(my_rank == 2){
    //     for(int i = 0; i < comm_size; ++i){
    //         std::cout << global_needed_heri_displ_arr[i] << std::endl; //displacements
    //     }
    // }
    // printf("\n");

    // if(my_rank == 2){
    //     for(int i = 0; i < comm_size; ++i){
    //         std::cout << all_local_needed_heri_sizes[i] << std::endl; //counts
    //     }
    // }
    // printf("\n");


    // std::cout << "Proc: " << my_rank << " has " << local_needed_heri_size << std::endl;

    // if(my_rank == 2){std::cout << all_local_needed_heri_sizes[0] + all_local_needed_heri_sizes[1] + all_local_needed_heri_sizes[2] << std::endl;}
    // exit(0);
    // Third, collect all local_needed_heri arrays to every process
    // std::vector<IT> local_needed_heriTEST = &local_needed_heri[0];
    MPI_Allgatherv(&(*local_needed_heri)[0],
                local_needed_heri_size,
                MPI_INT,
                global_needed_heri,
                all_local_needed_heri_sizes, //counts
                global_needed_heri_displ_arr, //displacements
                MPI_INT,
                MPI_COMM_WORLD);

    // exit(0);

    // Lastly, sort the global_needed_heri into "to_send_heri" 
    for(int from_proc = 1; from_proc < global_needed_heri_size; from_proc += 3){
        if(global_needed_heri[from_proc] == my_rank){
            to_send_heri->push_back(global_needed_heri[from_proc - 1]);
            to_send_heri->push_back(global_needed_heri[from_proc]);
            to_send_heri->push_back(global_needed_heri[from_proc + 1]);
        }
    }

    delete[] all_local_needed_heri_sizes;
    delete[] global_needed_heri_displ_arr;
    delete[] global_needed_heri;
}

// TODO: Best to pass pointers explicitly? Make consistent
template <typename VT, typename IT>
void communicate_halo_elements(std::vector<IT> &local_needed_heri, std::vector<IT> &to_send_heri, std::vector<VT> &local_x_scs_vec, const int *work_sharing_arr){//, int *commed_elems){
    /* The purpose of this function, is to allow each process to exchange it's proc-local "remote elements"
    with the other respective processes. For now, elements are sent one at a time. 
    Recall, tuple elements in needed_heri are formatted (proc to, proc from, global x idx). 
    Since this function is in the benchmark loop, need to do as little work possible, ideally.
    
    The global index WITH PADDING is unique, and used as the tag for communication. */

    // TODO: better to pass as args to function?
    int my_rank, comm_size;
    int rows_in_from_proc = work_sharing_arr[my_rank + 1] - work_sharing_arr[my_rank];
    int rows_in_to_proc, to_proc;
    int recieved_elems = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Status status;
    // ++(*commed_elems);

    // std::cout << *commed_elems << std::endl;
    // printf("%c", typeid(VT).name());
    // std::cout << typeid(VT).name() << std::endl;
    // std::cout << typeid(float).name() << std::endl;
    // std::cout << typeid(double) << std::endl;
    

    // IT local_x_idx = global_x_idx - work_sharing_arr[my_rank];
    // VT val_to_send;

    // // TODO: DRY
    // if(typeid(VT) == typeid(float)){
    //     // printf("I get inside\n");
    //     for(auto heri_tuple : local_needed_heri){
    //         // NOTE: assuming this is right for now, and moving on
    //         // std::cout << "Proc:" << my_rank << " needs x_global idx " << std::get<1>(heri_tuple) << " from " << std::get<0>(heri_tuple) << std::endl;

    //         /* Here, this proc would be doing the sending. 
    //         The value we send is the global, non-padded index of the x-vector,
    //         minus the arrpropriate amount of spaces to localize the index to the process. */
    //         to_proc = std::get<0>(heri_tuple);
    //         rows_in_to_proc = work_sharing_arr[to_proc + 1] - work_sharing_arr[to_proc];

    //         MPI_Recv(
    //             &local_x_scs_vec[rows_in_from_proc + recieved_elems],
    //             1,
    //             MPI_FLOAT,
    //             std::get<1>(heri_tuple),
    //             rows_in_from_proc + recieved_elems,
    //             MPI_COMM_WORLD,
    //             &status);

    //         // We place the recieved value in the first available location in local_x,
    //         // i.e. the first padded space available. From there, we increment the index */
    //         ++recieved_elems;
    //     }
    //     for(auto heri_tuple : to_send_heri){
    //         MPI_Send(
    //             &local_x_scs_vec[std::get<2>(heri_tuple) - work_sharing_arr[my_rank]],
    //             1,
    //             MPI_FLOAT,
    //             to_proc,
    //             rows_in_to_proc + recieved_elems + 1, // +1 to send to the idx after?
    //             MPI_COMM_WORLD);
    //     }
    // }
    
    // else if(typeid(VT) == typeid(double)){
    //     for(auto heri_tuple: needed_heri){
    //         // NOTE: assuming this is right for now, and moving on
    //         // std::cout << "Proc:" << my_rank << " needs x_global idx " << std::get<1>(heri_tuple) << " from " << std::get<0>(heri_tuple) << std::endl;

    //         // Here, this proc would be doing the sending
    //         if(my_rank == get<1>(heri_tuple)){
    //             MPI_Send(const void* buffer,
    //                 1,
    //                 MPI_DOUBLE,
    //                 get<0>(heri_tuple),
    //                 int tag,
    //                 MPI_COMM_WORLD);
    //         }
    //         // Here, this proc would be doing the recieving
    //         else if(my_rank == get<0>(heri_tuple)){
    //             MPI_Recv(void* buffer,
    //                 1,
    //                 MPI_DOUBLE,
    //                 get<1>(heri_tuple),
    //                 int tag,
    //                 MPI_COMM_WORLD,
    //                 MPI_Status* status);
    //         }
    //     }
    // }
}


// TODO: what do I return for this?
// NOTE: every process will return something...
template <typename VT, typename IT>
void bench_spmv_scs(
    const Config *config,
    const MtxData<VT, IT> &local_mtx,
    const int *work_sharing_arr,
    std::vector<VT> *total_y, // IDK if this is const
    DefaultValues<VT, IT> &defaults,
    const std::vector<VT> *x_in = nullptr)
{


    // TODO: More efficient just to deduce this from worksharingArr size?
    int comm_size, my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    log("allocate and place CPU matrices start\n");

    // BenchmarkResult r;
    // gathered y can be allocated here?
    ScsData<VT, IT> scs;

    log("allocate and place CPU matrices end\n");
    log("converting to scs format start\n");

    // IT x_col_upper = IT{}, x_col_lower = IT{};

    // TODO: fuse potentially
    convert_to_scs<VT, IT>(local_mtx, config->chunk_size, config->sigma, scs);
    // convert_to_scs<VT, IT>(mtx, config.chunk_size, config.sigma, scs, &x_col_upper, &x_col_lower);

    // std::vector<VT> total_x;
    // std::vector<VT> total_y(scs.n_rows);

    // std::vector<VT> local_x;

    log("converting to scs format end\n");

    // This y is only process local. Need an Allgather for each proc to have
    // all of the solution segments
    V<VT, IT> local_y_scs = V<VT, IT>(scs.n_rows_padded);
    std::uninitialized_fill_n(local_y_scs.data(), local_y_scs.n_rows, defaults.y);

    // TODO: How often is this allocated? Every swap, or only once?
    // Every processes needs counts array
    int *counts_arr = new int[comm_size];
    int *displ_arr_bk = new int[comm_size];

    // Set upper and lower bounds for local x vector
    // IT x_col_upper = *max_element(local_mtx.J.begin(), local_mtx.J.end());
    // IT x_col_lower = *min_element(local_mtx.J.begin(), local_mtx.J.end());

    // Alternatively, set the local x-vector rows to be the same rows
    // that the local mtx struct has (what to do in the case of non-sq matricies...?)
    // std::cout << work_sharing_arr[my_rank] << std::endl;
    // std::cout << work_sharing_arr[my_rank + 1] << std::endl;

    // exit(1);
    // IT local_first_row = work_sharing_arr[my_rank];
    // IT local_last_row = work_sharing_arr[my_rank + 1] - 1;
    // IT x_row_upper = local_mtx.I[local_first_row];
    // IT x_row_lower = local_mtx.I[local_last_row];

    IT x_row_lower = work_sharing_arr[my_rank];
    IT x_row_upper = work_sharing_arr[my_rank + 1] - 1;

    // std::cout << x_row_upper << std::endl;
    // std::cout << x_row_lower << std::endl;
    // exit(0);

    IT updated_col_idx, initial_col_idx = scs.col_idxs[0];

    // Shift local column indices
    for(int i = 0; i < scs.n_elements; ++i){
        updated_col_idx = scs.col_idxs[i] - initial_col_idx;

        if(updated_col_idx < 0){
            // padded case
            scs.col_idxs[i] = 0;    
        }
        else{
            scs.col_idxs[i] = updated_col_idx;
        }
    }

    // V<VT, IT> x_scs(scs.n_cols);
    // The local_x SHOULD only be as large as the number of rows in the local matrix
    V<VT, IT> local_x_scs(x_row_upper - x_row_lower + 1);

    // Boolean value in last arguement determines if x is random, or taken from default values
    // NOTE: may be important for swapping
    // Just replace with a standard vector?
    init_with_ptr_or_value(local_x_scs, local_x_scs.n_rows, x_in,
                            defaults.x, false);
    // init_with_ptr_or_value(x_scs, x_scs.n_rows, x_in,
    //                     defaults.x, false);

    // This vector is used to locate the non-padded elements of a collected/global x vector
    // for use in halo communication. gnp := global NO PADDING
    std::vector<IT> gnp_idx(local_x_scs.n_rows);
    for(int i = 0; i < local_x_scs.n_rows; ++i){
        gnp_idx[i] = work_sharing_arr[my_rank] + i;
    }

    // for(int i = 0; i < gnp_idx.size(); ++i){
    //     std::cout << "Proc: " << my_rank << ", gnp idx: " << gnp_idx[i] << std::endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(0);

    // The total x vector used in benchmarking
    // std::vector<VT> total_dummy_x(scs.n_rows);

    if(config->mode == "solver"){
        // TODO: make modifications for solver mode
        // the object "returned" will be the gathered results vector
    }
    else if(config->mode == "bench"){
        // TODO: make modifications for bench mode
        // the object returned will be the benchmark results
    }

    // x_out not used?
    // x_out = std::move(local_x_scs);
    // total_x = std::move(x_scs);
    // local_y.resize(scs.n_rows);

    // Really want these as standard vectors, not Vector custom object
    // TODO: How bad is this for scalability? Is there a way around this?
    std::vector<VT> local_x_scs_vec(local_x_scs.data(), local_x_scs.data() + local_x_scs.n_rows);
    std::vector<VT> local_y_scs_vec(local_y_scs.data(), local_y_scs.data() + local_y_scs.n_rows);


    // // What I really want here, is to be able to do all the swapping locally,
    // // and then collect only once before returning result
    // // (but then all processes have full y... is this what I want?)

    // The locations of the remote elements that an arbitrary process_k requires remain 
    // constant between iterations, and thus, can all be computed before entering the benchmark loop
    // for(int i = 0; i < scs.nnz; ++i)
    //     std::cout << scs.values.data()[i] << std::endl;
    // if(my_rank == 0){
    //     for(int j = 0; j < *nnz; ++j)
    //         // printf("%i\n", work_sharing_arr[j]);
    //         std::cout << scs.nnz << " ==? " << scs.values << std::endl;
    // }
    // exit(1);

    // heri := halo element row indices
    std::vector<IT> local_needed_heri;
    collect_local_needed_heri<VT, IT>(&local_needed_heri, local_mtx, work_sharing_arr);
    // "local_needed_heri" is all the halo elements that this process needs
    // "to_send_heri" are all halo elements that this process is to send
    // both are vectors, who's elements encode (proc to, proc from, global row idx) 3-tuples

    std::vector<IT> to_send_heri;
    collect_to_send_heri<VT, IT>(&to_send_heri, &local_needed_heri, local_mtx, work_sharing_arr);

    // if(my_rank == 0){
    //     for(int i = 1; i < to_send_heri.size(); i += 3){
    //         std::cout << "(TO: " << to_send_heri[i - 1] << ", FROM: " << to_send_heri[i] << 
    //         ", GLOB_IDX: " << to_send_heri[i + 1] << ")" << std::endl;
    //     }
    // }

    // exit(0);


    // for(auto heri_tuple : global_needed_heri){ // TODO: Would love to parallelize
    //     if(std::get<1>(heri_tuple)){
    //         to_send_heri->push_back(heri_tuple);
    //     }
    // }

    // if(my_rank == 0){
    //     for(int i = 0; i < comm_size + 1; ++i){
    //         std::cout << work_sharing_arr[i] << std::endl;
    //     }
    // }
    // exit(0);

    // if(my_rank == 2){
    //     for(auto tup : to_send_heri){
    //         std::cout << "Proc " << std::get<0>(tup) << " needs x_global idx " << 
    //         std::get<2>(tup) << " from Proc " << std::get<1>(tup) << std::endl;
    //     }
    // }

    exit(0);

    // if(my_rank == 2){
    //     for(int i = 0; i < local_x_scs_vec.size(); ++i){
    //         std::cout << local_x_scs_vec[i] << std::endl;
    //     }
    // }
    // printf("\n");

    // Pad the end of x-vector for incoming halo elements
    local_x_scs_vec.resize(local_x_scs.n_rows + local_needed_heri.size());

    // if(my_rank == 2){
    //     for(int i = 0; i < local_x_scs_vec.size(); ++i){
    //         std::cout << local_x_scs_vec[i] << std::endl;
    //     }
    // }

    // }
    // std::cout << local_needed_heri.size() << std::endl;
    // std::cout << local_x_scs_vec.size() << " and " << local_mtx.n_rows << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);

    for (int i = 0; i < config->n_repetitions; ++i)
    {    
        // Proc-local counter for number of already communicated halo elements
        // int commed_elems = 0;
        // MPI_Barrier(MPI_COMM_WORLD); // temporary
        communicate_halo_elements<VT, IT>(local_needed_heri, to_send_heri, local_x_scs_vec, work_sharing_arr);//, &commed_elems);
        // exit(0);

        // if(local_x_scs.n_rows < local_mtx.n_cols){
        //     local_x_scs_vec.resize(local_mtx.n_cols)
        // }
        // std::cout << local_x_scs.n_rows << std::endl;
        // exit(1);

        // communicate_halo_elements(local_x_scs)

        // Do the actual multiplication
        // TODO: store .data() members locally as const
        spmv_omp_scs_c<VT, IT>( scs.C, scs.n_chunks, scs.chunk_ptrs.data(), 
                                scs.chunk_lengths.data(),scs.col_idxs.data(), 
                                scs.values.data(), &(local_x_scs_vec)[0], &(local_y_scs_vec)[0]);
                                // change last 2 arguements to std::vectors, not Vectors

        // swap pointer
        // only using full x_scs not for size of vector concerns 

        // TODO: double check sizes here!!
        // std::swap(total_dummy_x, local_y_scs);
        std::swap(local_x_scs_vec, local_y_scs_vec);
        // std::cout << "repetition: " << i << std::endl;
    }

    // TODO: use a pragma parallel for?
    // Reformat proc-local result vectors. Only take the useful (non-padded) elements 
    // from the scs formatted local_y_scs, and assign to local_y
    std::vector<VT> local_y;
    local_y.resize(scs.n_rows);

    for (int i = 0; i < scs.old_to_new_idx.n_rows; ++i)
    {
        local_y[i] = local_y_scs_vec[scs.old_to_new_idx[i]];
    }

    // NOTE: only for collecting local results.
    // i.e. wont be apart of actual loop after halo comm implemented
    for (int i = 0; i < comm_size; ++i)
    {
        counts_arr[i] = work_sharing_arr[i + 1] - work_sharing_arr[i];
        displ_arr_bk[i] = work_sharing_arr[i];
        std::cout << counts_arr[i] << ", " << displ_arr_bk[i] << std::endl;    
    }

    // std::cout << local_y.size() << std::endl;
    // std::cout << &local_y[0] << std::endl;
    // std::cout << &total_y[0] << std::endl;

    // Can use Allgatherv to collect results from each proc to y_total,
    // since messages are of varying size
    // if (typeid(VT) == typeid(double))
    // {
    //     MPI_Allgatherv(&local_y[0],
    //                 local_y.size(),
    //                 MPI_DOUBLE,
    //                 &(*total_y)[0],
    //                 counts_arr,
    //                 displ_arr_bk,
    //                 MPI_DOUBLE,
    //                 MPI_COMM_WORLD);
    // }
    // else if (typeid(VT) == typeid(float))
    // {
    //     MPI_Allgatherv(&local_y[0],
    //                 local_y.size(),
    //                 MPI_FLOAT,
    //                 &(*total_y)[0],
    //                 counts_arr,
    //                 displ_arr_bk,
    //                 MPI_FLOAT,
    //                 MPI_COMM_WORLD);
    // }

    // total_y = std::move(local_x_scs);    // x_out = std::move(local_x_scs);

    // std::cout << "\n" << "Proc: " << my_rank << " | Original x_vec length: " << mtx.n_cols
    // << ", Shortened x_vec length: " << x_col_upper - x_col_lower + 1 << std::endl;

    delete[] counts_arr;
    delete[] displ_arr_bk;

    // return total_y;
}

/**
 * Benchmark the OpenMP/GPU spmv kernel with a matrix of dimensions \p n_rows x
 * \p n_cols.  If \p mmr is not NULL, then the matrix is read via the provided
 * MatrixMarketReader from file.
 */
// template <typename VT, typename IT>
// static BenchmarkResult
//  bench_spmv(const std::string &kernel_name,
//            const Config &config,
//            const Kernel::entry_t &k_entry,
//            const MtxData<VT, IT> &mtx,
//            const int *work_sharing_arr,
//            std::vector<VT> *y_total, // IDK if this is const
//            DefaultValues<VT, IT> *defaults = nullptr,
//            const std::vector<VT> *x_in = nullptr,
//            std::vector<VT> *y_out_opt = nullptr)
// {

//     BenchmarkResult r;

//     // How are these involved with swapping?
//     std::vector<VT> y_out;
//     std::vector<VT> x_out;

//     DefaultValues<VT, IT> default_values;

//     if (!defaults)
//     {
//         defaults = &default_values;
//     }

//     switch (k_entry.format)
//     {
//     case MatrixFormat::Csr:
//         r = bench_spmv_csr<VT, IT>(config,
//                                    mtx,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;
//     case MatrixFormat::EllRm:
//     case MatrixFormat::EllCm:
//         r = bench_spmv_ell<VT, IT>(config,
//                                    mtx,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;
//     case MatrixFormat::SellCSigma:
//         r = bench_spmv_scs<VT, IT>(config,
//                                    mtx,
//                                    work_sharing_arr, y_total,
//                                    k_entry, *defaults,
//                                    x_out, y_out, x_in);
//         break;
//     default:
//         fprintf(stderr, "ERROR: SpMV format for kernel %s is not implemented.\n", kernel_name.c_str());
//         return r;
//     }

//     if (y_out_opt)
//         *y_out_opt = std::move(y_out);

//     // if (print_matrices) {
//     //     printf("Matrices for kernel: %s\n", kernel_name.c_str());
//     //     printf("A, is_col_major: %d\n", A.is_col_major);
//     //     print(A);
//     //     printf("b\n");
//     //     print(b);
//     //     printf("x\n");
//     //     print(x);
//     // }

//     return r;
// }

/**
 * @brief Return the file base name without an extension, empty if base name
 *        cannot be extracted.
 * @param file_name The file name to extract the base name from.
 * @return the base name of the file or an empty string if it cannot be
 *         extracted.
 */
static std::string
file_base_name(const char *file_name)
{
    if (file_name == nullptr)
    {
        return std::string{};
    }

    std::string file_path(file_name);
    std::string file;

    size_t pos_slash = file_path.rfind('/');

    if (pos_slash == file_path.npos)
    {
        file = std::move(file_path);
    }
    else
    {
        file = file_path.substr(pos_slash + 1);
    }

    size_t pos_dot = file.rfind('.');

    if (pos_dot == file.npos)
    {
        return file;
    }
    else
    {
        return file.substr(0, pos_dot);
    }
}

static void
print_histogram(const char *name, const Histogram &hist)
{
    size_t n_buckets = hist.bucket_counts().size();

    for (size_t i = 0; i < n_buckets; ++i)
    {
        printf("%-20s %3lu  %7ld  %7lu\n",
               name,
               i,
               hist.bucket_upper_bounds()[i],
               hist.bucket_counts()[i]);
    }
}

template <typename T>
static std::string
to_string(const Statistics<T> &stats)
{
    std::stringstream stream;

    stream << "avg: " << std::scientific << std::setprecision(2) << stats.avg
           << " s: " << std::scientific << std::setprecision(2) << stats.std_dev
           << " cv: " << std::scientific << std::setprecision(2) << stats.cv
           << " min: " << std::scientific << std::setprecision(2) << stats.min
           << " median: " << std::scientific << std::setprecision(2) << stats.median
           << " max: " << std::scientific << std::setprecision(2) << stats.max;

    return stream.str();
}

template <typename T>
static void
print_matrix_statistics(
    const MatrixStats<T> &matrix_stats,
    const std::string &matrix_name)
{

    printf("##mstats %19s  %7s %7s  %9s  %5s %5s  %8s  %8s  %9s  %6s %6s\n",
           "name", "rows", "cols", "nnz", "nzr", "nzc", "maxrow",
           "density", "bandwidth",
           "sym", "sorted");

    const char *name = "unknown";

    if (!matrix_name.empty())
    {
        name = matrix_name.c_str();
    }

    printf("#mstats %-20s  %7ld %7ld  %9ld  %5.2f %5.2f  %8.2e  %8.2e  %9lu  %6d %6d\n",
           name,
           matrix_stats.n_rows, matrix_stats.n_cols,
           matrix_stats.nnz,
           matrix_stats.row_lengths.avg,
           matrix_stats.col_lengths.avg,
           matrix_stats.row_lengths.max,
           matrix_stats.densitiy,
           (uint64_t)matrix_stats.bandwidths.max,
           matrix_stats.is_symmetric,
           matrix_stats.is_sorted);

    printf("#mstats-nzr %-20s %s\n", name, to_string(matrix_stats.row_lengths).c_str());
    printf("#mstats-nzc %-20s %s\n", name, to_string(matrix_stats.col_lengths).c_str());
    printf("#mstats-bw  %-20s %s\n", name, to_string(matrix_stats.bandwidths).c_str());

    print_histogram("#mstats-rows", matrix_stats.row_lengths.hist);
    print_histogram("#mstats-cols", matrix_stats.col_lengths.hist);
    print_histogram("#mstats-bws", matrix_stats.bandwidths.hist);
}


// #include "test.cpp"

// static void
// usage()
// {
//     fprintf(stderr, "Usage:\n");
//     fprintf(stderr,
//             "  spmv-<omp|gpu> <martix-market-filename> [[kernel|all] [-sp|-dp|-all] [-c C] [-s Sigma] [-nzr NZR]]\n");
//     fprintf(stderr,
//             "  spmv-<omp|gpu> list\n");
//     fprintf(stderr,
//             "  spmv-<omp|gpu> test\n");
// }

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

// template <typename IT, VT>
// struct Results
// {
//     std::vector<VT> final_y;
//     int 
// }

template <typename IT>
IT get_index(std::vector<IT> v, int K)
{
    /*
    Gets the index of the first instance of the element K in the vector v.
    */
    auto it = find(v.begin(), v.end(), K);

    // If element was found
    if (it != v.end())
    {

        // calculating the index
        // of K
        int index = it - v.begin();
        return index;
    }
    else
    {
        // If the element is not
        // present in the vector
        return -1; // TODO: implement better error
    }
}

template <typename VT, typename IT>
void seg_work_sharing_arr(const MtxData<VT, IT> &mtx, int *work_sharing_arr, std::string seg_method, int comm_size)
{
    work_sharing_arr[0] = 0;

    int segment;

    if ("seg-rows" == seg_method)
    {
        int rowsPerProc;

        // Evenly split the number of rows
        rowsPerProc = mtx.n_rows / comm_size;

        // Segment rows to work on via. array
        for (segment = 1; segment < comm_size + 1; ++segment)
        {
            // Can only do this because of "constant sized" segments
            work_sharing_arr[segment] = segment * rowsPerProc;
            if (segment == comm_size)
            {
                // Set the last element to point to directly after the final row
                // (takes care of remainder rows)
                work_sharing_arr[comm_size] = mtx.I[mtx.nnz - 1] + 1;
            }
        }
    }
    else if ("seg-nnz" == seg_method)
    {
        int nnzPerProc; //, remainderNnz;

        // Split the number of rows based on non zeros
        nnzPerProc = mtx.nnz / comm_size;
        // remainderNnz = mtx.nnz % nnzPerProc;

        int global_ctr, local_ctr;
        segment = 1;
        local_ctr = 0;

        // Segment rows to work on via. array
        for (global_ctr = 0; global_ctr < mtx.nnz; ++global_ctr)
        {
            if (local_ctr == nnzPerProc)
            {
                // Assign rest of the current row to this segment
                work_sharing_arr[segment] = mtx.I[global_ctr] + 1;
                ++segment;
                local_ctr = 0;
                continue;
            }
            ++local_ctr;
        }
        // Set the last element to point to directly after the final row
        // (takes care of remainder rows)
        work_sharing_arr[comm_size] = mtx.I[mtx.nnz - 1] + 1;
    }
}

template <typename VT, typename IT>
void segMtxStruct(const MtxData<VT, IT> &mtx, std::vector<IT> *local_I, std::vector<IT> *local_J, std::vector<VT> *local_vals, int *work_sharing_arr, int loop_rank)
{
    int start_idx, run_idx, finish_idx;
    int next_row;

    // Assign rows, columns, and values to process local vectors
    for (int row = work_sharing_arr[loop_rank]; row < work_sharing_arr[loop_rank + 1]; ++row)
    {
        next_row = row + 1;

        // Return the first instance of that row present in mtx.
        start_idx = get_index<IT>(mtx.I, row);

        // once we have the index of the first instance of the row,
        // we calculate the index of the first instance of the next row
        if (next_row != mtx.n_rows)
        {
            finish_idx = get_index<IT>(mtx.I, next_row);
        }
        else
        {
            // for the "last row" case, just set finish_idx to the number of non zeros in mtx
            finish_idx = mtx.nnz;
        }
        run_idx = start_idx;

        // This do-while loop will go "across the rows", basically filling the process local vectors
        // TODO: is this better than a while loop here?
        do
        {
            local_I->push_back(mtx.I[run_idx]);
            local_J->push_back(mtx.J[run_idx]);
            local_vals->push_back(mtx.values[run_idx]);
            ++run_idx;
        } while (run_idx != finish_idx);
    }
}

void define_bookkeeping_type(MtxDataBookkeeping<long int> *send_bk, MPI_Datatype *bk_type)
{

    // Declare and define MPI Datatype
    int block_length_arr[2];
    MPI_Aint displ_arr_bk[2], first_address, second_address;

    MPI_Datatype type_arr[2];
    type_arr[0] = MPI_LONG;
    type_arr[1] = MPI_CXX_BOOL;
    block_length_arr[0] = 3; // using 3 int elements
    block_length_arr[1] = 2; // and 2 bool elements
    MPI_Get_address(&send_bk->n_rows, &first_address);
    MPI_Get_address(&send_bk->is_sorted, &second_address);

    displ_arr_bk[0] = (MPI_Aint)0; // calculate displacements from addresses
    displ_arr_bk[1] = MPI_Aint_diff(second_address, first_address);
    MPI_Type_create_struct(2, block_length_arr, displ_arr_bk, type_arr, bk_type);
    MPI_Type_commit(bk_type);
}

template <typename VT, typename IT, typename ST>
void seg_and_send_data(MtxData<VT, IT> &local_mtx, Config config, std::string seg_method, std::string file_name_str, int *work_sharing_arr, int my_rank, int comm_size)
{

    // Declare functions to be used locally
    // void segwork_sharing_arr(const MtxData<VT, IT>, int *, const char *, int);
    // void segMtxStruct(const MtxData<VT, IT>, std::vector<IT> *, std::vector<IT> *, std::vector<VT> *, int *, int);

    MPI_Status status_bk, status_cols, status_rows, status_vals;

    MtxDataBookkeeping<ST> send_bk, recv_bk;
    MPI_Datatype bk_type;

    define_bookkeeping_type(&send_bk, &bk_type);

    int msg_length;

    if (my_rank == 0)
    {
        // NOTE: Matrix will be read in as SORTED by default
        // Only root proc will read entire matrix
        MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name_str.c_str(), config.sort_matrix);

        // Segment global row pointers, and place into an array
        // int *work_sharing_arr = new int[comm_size + 1];
        seg_work_sharing_arr<VT, IT>(mtx, work_sharing_arr, seg_method, comm_size);

        // Eventhough we're iterting through the ranks, this loop is
        // (in the present implementation) executing sequentially on the root proc
        for (int loop_rank = 0; loop_rank < comm_size; ++loop_rank)
        { // NOTE: This loop assumes we're using all ranks 0 -> comm_size-1
            std::vector<IT> local_I;
            std::vector<IT> local_J;
            std::vector<VT> local_vals;

            // Assign rows, columns, and values to process local vectors
            segMtxStruct<VT, IT>(mtx, &local_I, &local_J, &local_vals, work_sharing_arr, loop_rank);

            // Count the number of rows in each processes
            int local_row_cnt = std::set<IT>(local_I.begin(), local_I.end()).size();

            // Here, we segment data for the root process
            if (loop_rank == 0)
            {
                local_mtx = {
                    local_row_cnt,
                    mtx.n_cols,
                    local_vals.size(),
                    config.sort_matrix,
                    0,          // NOTE: These "sub matricies" will (almost) never be symmetric
                    local_I, // should work as both local and global row ptr
                    local_J,
                    local_vals};
            }
            // Here, we segment and send data to another proc
            else
            {
                send_bk = {
                    local_row_cnt,
                    mtx.n_cols, // TODO: Actually constant, do dont need to send to each proc
                    local_vals.size(),
                    config.sort_matrix,
                    0};

                // First, send BK struct
                MPI_Send(&send_bk, 1, bk_type, loop_rank, 99, MPI_COMM_WORLD);

                // Next, send three arrays
                MPI_Send(&local_I[0], local_I.size(), MPI_INT, loop_rank, 42, MPI_COMM_WORLD);
                MPI_Send(&local_J[0], local_J.size(), MPI_INT, loop_rank, 43, MPI_COMM_WORLD);
                if (typeid(VT) == typeid(double))
                {
                    MPI_Send(&local_vals[0], local_vals.size(), MPI_DOUBLE, loop_rank, 44, MPI_COMM_WORLD);
                }
                else if (typeid(VT) == typeid(float))
                {
                    MPI_Send(&local_vals[0], local_vals.size(), MPI_FLOAT, loop_rank, 44, MPI_COMM_WORLD);
                }
            }
        }
        // delete[] work_sharing_arr;
    }
    else if (my_rank != 0)
    {
        // First, recieve BK struct
        MPI_Recv(&recv_bk, 1, bk_type, 0, 99, MPI_COMM_WORLD, &status_bk);

        // Next, allocate space for incoming arrays
        msg_length = recv_bk.nnz;
        IT *recv_buf_global_row_coords = new IT[msg_length];
        IT *recv_buf_col_coords = new IT[msg_length];
        VT *recv_buf_vals = new VT[msg_length];

        // Next, recieve 3 arrays that we've allocated space for on local proc
        MPI_Recv(recv_buf_global_row_coords, msg_length, MPI_INT, 0, 42, MPI_COMM_WORLD, &status_rows);
        MPI_Recv(recv_buf_col_coords, msg_length, MPI_INT, 0, 43, MPI_COMM_WORLD, &status_cols);
        if (typeid(VT) == typeid(double))
        {
            MPI_Recv(recv_buf_vals, msg_length, MPI_DOUBLE, 0, 44, MPI_COMM_WORLD, &status_vals);
        }
        else if (typeid(VT) == typeid(float))
        {
            MPI_Recv(recv_buf_vals, msg_length, MPI_FLOAT, 0, 44, MPI_COMM_WORLD, &status_vals);
        }
        // TODO: Just how bad is this?... Are we copying array -> vector?
        std::vector<IT> global_rows_vec(recv_buf_global_row_coords, recv_buf_global_row_coords + msg_length);
        std::vector<IT> cols_vec(recv_buf_col_coords, recv_buf_col_coords + msg_length);
        std::vector<VT> vals_vec(recv_buf_vals, recv_buf_vals + msg_length);

        local_mtx = {
            recv_bk.n_rows,
            recv_bk.n_cols,
            recv_bk.nnz,
            recv_bk.is_sorted,
            recv_bk.is_symmetric,
            global_rows_vec,
            cols_vec,
            vals_vec};

        // Notice, recv_buf_global_row_coords not deleted, because each process needs to know
        // it's global row pointer for reconstruction later
        delete[] recv_buf_global_row_coords;
        delete[] recv_buf_col_coords;
        delete[] recv_buf_vals;
    }

    // Each process exchanges it's global row ptrs for local row ptrs
    IT first_global_row = local_mtx.I[0];
    IT *global_row_coords = new IT[local_mtx.nnz];
    IT *local_row_coords = new IT[local_mtx.nnz];

    for (int nz = 0; nz < local_mtx.nnz; ++nz)
    {
        // save proc's global row ptr
        global_row_coords[nz] = local_row_coords[nz];

        // subtract first pointer from the rest, to make them process local
        local_row_coords[nz] = local_mtx.I[nz] - first_global_row;
    }

    std::vector<IT> loc_rows_vec(local_row_coords, local_row_coords + local_mtx.nnz);

    // assign local row ptrs to struct
    local_mtx.I = loc_rows_vec;

    // Broadcast work sharing array to other processes
    MPI_Bcast(work_sharing_arr,
              comm_size + 1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    // NOTE: what possibilities are there to use global row coords later?
    delete[] global_row_coords;
    delete[] local_row_coords;

    MPI_Type_free(&bk_type);
}

template <typename VT, typename IT>
void check_if_result_valid(const char *file_name, std::vector<VT> *y_total, const std::string name, bool sort_matrix)
{
    DefaultValues<VT, IT> defaults;

    // Root proc reads all of mtx
    MtxData<VT, IT> mtx = read_mtx_data<VT, IT>(file_name, sort_matrix);

    std::vector<VT> x_total(mtx.n_cols);
    // std::uninitialized_fill_n(x_total.data(), x_total.n_rows, idk.y);

    // recreate_x_total()

    // TODO: Only works since seed is same. Not flexible to swapping.
    init_std_vec_with_ptr_or_value<VT>(x_total, mtx.n_cols, nullptr, defaults.x);

    // bool is_result_valid = spmv_verify<VT, IT>(name, mtx, x_total, *y_total);

    // std::cout << result_valid << std::endl;

    // TODO: come back to validity checking later
    // if (is_result_valid)
    // {
    //     printf("Results valid.\n");
    // }
    // else
    // {
    //     printf("Results NOT valid.\n");
    // }
}

template <typename VT, typename IT>
void compute_result(std::string file_name, std::string seg_method, Config config)
{
    // BenchmarkResult result;

    // MatrixStats<double> matrix_stats;
    // bool matrix_stats_computed = false;

    // Initialize MPI variables
    int my_rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Declare struct on each process
    MtxData<VT, IT> local_mtx;

    // Allocate space for work sharing array. Is populated in seg_and_send_data function
    IT *work_sharing_arr = new IT[comm_size + 1];

    seg_and_send_data<VT, IT, ST>(local_mtx, config, seg_method, file_name, work_sharing_arr, my_rank, comm_size);

    // if (!matrix_stats_computed)
    // {
    //     matrix_stats = get_matrix_stats(local_mtx);
    //     matrix_stats_computed = true;
    // }

    // Each process must allocate space for total y vector
    // Will these just be the same, i.e. dont need to return anything?
    std::vector<VT> total_y(work_sharing_arr[comm_size]);
    std::vector<VT> result(work_sharing_arr[comm_size]);

    // result = bench_spmv<VT, IT>(name, config, k_entry, local_mtx, work_sharing_arr, &y_total);
//  bench_spmv(const std::string &kernel_name,
//            const Config &config,
//            const Kernel::entry_t &k_entry,
//            const MtxData<VT, IT> &mtx,
//            const int *work_sharing_arr,
//            std::vector<VT> *y_total, // IDK if this is const
//            DefaultValues<VT, IT> *defaults = nullptr,
//            const std::vector<VT> *x_in = nullptr,
//            std::vector<VT> *y_out_opt = nullptr)


    // TODO: whats the best way to handle defaults? Just throw in config struct?
    DefaultValues<VT, IT> default_values;
        // if (!defaults) {
        //     defaults = &default_values;
        // }

    bench_spmv_scs<VT, IT>(&config,
                                   local_mtx,
                                   work_sharing_arr, 
                                   &total_y,
                                   default_values);

    for (auto res: total_y)
        std::cout << res << std::endl;

    delete[] work_sharing_arr;

    // Every process prints it's mtx-local statistics
    // if (print_proc_local_stats)
    // {
    //     print_results(print_list, name, matrix_stats, result, n_cpu_threads, print_details);
    // }
    // if (config.verify_result)
    // {
    //     // But have root proc check results, because all processes have the same y_total
    //     if (my_rank == 0)
    //     {
    //         check_if_result_valid<VT, IT>(file_name, &y_total, name, config.sort_matrix);
    //     }
    // }
}

void verifyAndAssignInputs(int argc, char *argv[], std::string &file_name_str, std::string &seg_method, std::string &value_type, bool *random_init_x, Config *config){
    if (argc < 2){
        fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
            "options [defaults]: -c[%li], -s[%li], -rev[%li], -rand-x[%i], -sp/dp[%s], -seg-nnz/seg-rows[%s], -bench/solver[%s]\n",
            argv[0], config->chunk_size, config->sigma, config->n_repetitions, *random_init_x, value_type.c_str(), seg_method.c_str(), config->mode);
        exit(1);
    }

    file_name_str = argv[1];

    int args_start_index = 2;
    for (int i = args_start_index; i < argc; ++i){
        std::string arg = argv[i];
        if (arg == "-c")
        {
            config->chunk_size = atoi(argv[++i]);

            if (config->chunk_size < 1)
            {
                fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-s")
        {

            config->sigma = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->sigma < 1)
            {
                fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rev")
        {
            config->n_repetitions = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->n_repetitions < 1)
            {
                fprintf(stderr, "ERROR: revisions must be >= 1.\n");
                exit(1);
            }
        }
        else if (arg == "-rand-x")
        {
            *random_init_x = true;
        }
        else if (arg == "-dp")
        {
            value_type = "dp";
        }
        else if (arg == "-sp")
        {
            value_type = "sp";
        }
        else if (arg == "-seg-rows")
        {
            seg_method = "seg-rows";
        }
        else if (arg == "-seg-nnz")
        {
            seg_method = "seg-nnz";
        }
        else if (arg == "-bench")
        {
            config->mode = "bench";
        }
        else if (arg == "-solver")
        {
            config->mode = "solver";
        }
        else
        {
            fprintf(stderr, "ERROR: unknown argument.\n");
            exit(1);
        }
    }
    
    if (config->sigma > config->chunk_size){
            fprintf(stderr, "ERROR: sigma must be smaller than chunk size.\n");
            exit(1);
    }
}




int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    Config config;
    std::string file_name_str{};

    // Set defaults for cl inputs
    std::string seg_method{"seg-rows"};
    // std::string kernel_to_benchmark{"csr"}; still needed?
    std::string value_type = {"dp"};
    int C = config.chunk_size;
    int sigma = config.sigma;
    int revisions = config.n_repetitions;
    bool random_init_x = false;

    MARKER_INIT();

    verifyAndAssignInputs(argc, argv, file_name_str, seg_method, value_type, &random_init_x, &config);
    // std::cout << seg_method << " " << typeid(seg_method).name() << std::endl;
    // exit(1);

    if (value_type == "sp" )
    {
        compute_result<float, int>(file_name_str, seg_method, config);
    }
    else if (value_type == "dp")
    {
        compute_result<double, int>(file_name_str, seg_method, config);
    }
    
    log("benchmarking kernel: scs end\n");

    MPI_Finalize();

    log("main end\n");

    MARKER_DEINIT();

    return 0;
}

#ifndef UTILITIES
#define UTILITIES

#include "classes_structs.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef USE_CUSPARSE
#include <cusparse.h> 
#endif

#include <cstdarg>
#include <random>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <float.h>
#include <math.h>

#ifdef USE_SCAMAC
#include "scamac.h"
#endif

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

#ifdef USE_MPI
// TODO: Need to define your own MPI datatype for HALF
template<typename T>
MPI_Datatype get_mpi_type() = delete;

#ifdef HAVE_HALF_MATH
// TODO
template<>
MPI_Datatype get_mpi_type<_Float16>() {
  return MPI_BYTE;
}
#endif

template<>
MPI_Datatype get_mpi_type<float>() {
  return MPI_FLOAT;
}
template<>
MPI_Datatype get_mpi_type<double>() {
  return MPI_DOUBLE;
}
template<>
MPI_Datatype get_mpi_type<std::complex<float> >() {
  return MPI_C_FLOAT_COMPLEX;
}
template<>
MPI_Datatype get_mpi_type<std::complex<double> >() {
  return MPI_C_DOUBLE_COMPLEX;
}
#endif

template <typename VT>
void print_vector(const std::string &name,
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

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_add_overflow(T a, T b)
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
bool will_add_overflow(T a, T b)
{
    return std::numeric_limits<T>::max() - a < b;
}

template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value,
              bool>::type = true>
bool will_mult_overflow(T a, T b)
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
bool will_mult_overflow(T a, T b)
{
    if (a == 0 || b == 0)
    {
        return false;
    }

    return std::numeric_limits<T>::max() / a < b;
}

// std::tuple<std::string, uint64_t>
// type_info_from_type_index(const std::type_index &ti)
// {
//     static std::unordered_map<std::type_index, std::tuple<std::string, uint64_t>> type_map = {
//         {std::type_index(typeid(double)), std::make_tuple("dp", sizeof(double))},
//         {std::type_index(typeid(float)), std::make_tuple("sp", sizeof(float))},

//         {std::type_index(typeid(int)), std::make_tuple("int", sizeof(int))},
//         {std::type_index(typeid(long)), std::make_tuple("long", sizeof(long))},
//         {std::type_index(typeid(int32_t)), std::make_tuple("int32_t", sizeof(int32_t))},
//         {std::type_index(typeid(int64_t)), std::make_tuple("int64_t", sizeof(int64_t))},

//         {std::type_index(typeid(unsigned int)), std::make_tuple("uint", sizeof(unsigned int))},
//         {std::type_index(typeid(unsigned long)), std::make_tuple("ulong", sizeof(unsigned long))},
//         {std::type_index(typeid(uint32_t)), std::make_tuple("uint32_t", sizeof(uint32_t))},
//         {std::type_index(typeid(uint64_t)), std::make_tuple("uint64_t", sizeof(uint64_t))}};

//     auto it = type_map.find(ti);

//     if (it == type_map.end())
//     {
//         return std::make_tuple(std::string{"unknown"}, uint64_t{0});
//     }

//     return it->second;
// }

// std::string
// type_name_from_type_index(const std::type_index &ti)
// {
//     return std::get<0>(type_info_from_type_index(ti));
// }

// template <typename T>
// std::string
// type_name_from_type()
// {
//     return type_name_from_type_index(std::type_index(typeid(T)));
// }

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
struct Statistics<T>
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
MatrixStats<double>
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

/**
 * Compute maximum number of elements in a row.
 * \p num_rows: Number of rows.
 * \p nnz: Number of non-zeros, also number of elements in \p row_indices.
 * \p row_indices: Array with row indices.
 */
template <typename IT>
IT calculate_max_nnz_per_row(
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

template <typename IT>
void
convert_idxs_to_ptrs(const std::vector<IT> &idxs,
                    std::vector<IT> &ptrs)
{
    std::fill(ptrs.data(), ptrs.data() + ptrs.size(), 0);

    for (const auto idx : idxs)
    {
        if (idx + 1 < ptrs.size())
        {
            ++ptrs[idx + 1];
        }
    }

    std::partial_sum(ptrs.data(), ptrs.data() + ptrs.size(), ptrs.data());
}

/**
 * Compare vectors \p reference and \p actual.  Return the no. of elements that
 * differ.
 */
template <typename VT>
ST compare_arrays(const VT *reference,
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

template <typename VT>
bool spmv_verify(
    const VT *y_ref,
    const VT *y_actual,
    const ST n,
    bool verbose)
{
    VT max_rel_error_found{};

    ST error_counter =
        compare_arrays(
            y_ref, y_actual, n,
            verbose,
            max_rel_error<VT>::value,
            max_rel_error_found);

    if (error_counter > 0)
    {
        // TODO: fix reported name and sizes.
        fprintf(stderr,
                "WARNING: spmv kernel %s (fp size %lu, idx size %lu) is incorrect, "
                "relative error > %e for %ld/%ld elements. Max found rel error %e.\n",
                "", sizeof(VT), 0ul,
                max_rel_error<VT>::value,
                (long)error_counter, (long)n,
                max_rel_error_found);
    }

    return error_counter == 0;
}

template <typename VT, typename IT>
bool spmv_verify(const std::string *matrix_format,
            const MtxData<VT, IT> *mtx,
            const std::vector<VT> &x,
            const std::vector<VT> &y_actual)
{
    std::vector<VT> y_ref(mtx->n_rows);

    ST nnz = mtx->nnz;
    if (mtx->I.size() != mtx->J.size() || mtx->I.size() != mtx->values.size())
    {
        fprintf(stderr, "ERROR: %s:%d sizes of rows, cols, and values differ.\n", __FILE__, __LINE__);
        exit(1);
    }

    for (ST i = 0; i < nnz; ++i)
    {
        y_ref[mtx->I[i]] += mtx->values[i] * x[mtx->J[i]];
    }

    return spmv_verify(y_ref.data(), y_actual.data(),
                       y_actual.size(), /*verbose*/ true);
}

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

/**
 * @brief Return the file base name without an extension, empty if base name
 *        cannot be extracted.
 * @param file_name The file name to extract the base name from.
 * @return the base name of the file or an empty string if it cannot be
 *         extracted.
 */
std::string file_base_name(const char *file_name)
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

void print_histogram(const char *name, const Histogram &hist)
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
std::string to_string(const Statistics<T> &stats)
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
void print_matrix_statistics(
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

template <typename VT>
void random_init(
    Config *config,
    VT *begin, 
    VT *end
)
{
    int my_rank;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    srand(time(NULL) + my_rank);
#else
    srand(time(NULL));
#endif

    std::mt19937 engine;

    // if (!g_same_seed_for_every_vector)
    // {
    //     std::random_device rnd_device;
    //     engine.seed(rnd_device());
    // }

    std::uniform_real_distribution<double> dist(config->matrix_min, config->matrix_max);

    for (VT *it = begin; it != end; ++it)
    {
        // *it = ((VT) rand() / ((VT) RAND_MAX)) + 1;
        *it = dist(engine);
    }
}

template <typename VT>
void init_std_vec_with_ptr_or_value(
    Config *config,
    std::vector<VT> &x,
    ST n_x,
    int n_rows,
    int n_rows_padded,
    VT default_value,
    char init_with_random_numbers,
    const std::vector<VT> *x_in = nullptr
)
{
    if (init_with_random_numbers == '0' || init_with_random_numbers == 'm')
    {
        if (x_in) // NOTE: not used right now
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
        // Randomly initialize between max and min value of matrix
        random_init(config, &(*x.begin()), &(*x.end()));
    }

    // Insert zeros into halo element row index padding. 
    // Nice for debugging block vector stuff later
    int vec_idx = 0;
    for (ST i = 0; i < n_x; ++i)
    {

#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
        if(vec_idx >= n_rows){
            x[i] = VT{};
        }

        ++vec_idx;
        // NOTE: Hmmm. Not quite sure about this...

        if(vec_idx == n_rows_padded){
            vec_idx = 0;
        }
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
        if(vec_idx >= (n_rows * config->block_vec_size)){
            x[i] = VT{};
        }

        ++vec_idx;
#endif
    }
}

void cli_options_messge(
    int argc, 
    char *argv[],
    std::string *seg_method,
    std::string *value_type,
    Config *config){
    fprintf(stderr, "Usage: %s <martix-market-filename/SCAMAC-config> <kernel-format> [options]\n " 
        "options [defaults] (description): \n" \
        "-block_vec_size [%i] (int: width of block vectors for SpMMV) \n" \
        "-c [%li] (int: chunk size (required for scs)) \n" \
        "-s [%li] (int: sigma (required for scs)) \n" \
        "-rev [%li] (int: number of back-to-back revisions to perform) \n" \
        "-rand_x [%c] (0/1: random x vector option) \n" \
        "-dp / sp / hp / ap[dp_sp] / ap[dp_hp] / ap[sp_hp] / ap[dp_sp_hp] [%s] (numerical precision of matrix data) \n" \
        "-seg_metis / seg_nnz / seg_rows [%s] (global matrix partitioning for MPI) \n" \
        "-validate [%i] (0/1: check result against MKL option) \n" \
        "-verbose [%i] (0/1: verbose validation of results) \n" \
        "-mode [%c] ('s'/'b': either in solve mode or bench mode) \n" \
        "-bench_time [%g] (float: minimum number of seconds for SpMV benchmark) \n" \
        "-ba_synch [%i] (0/1: synch processes each benchmark loop) \n" \
        "-comm_halos [%i] (0/1: communicate halo elements each benchmark loop) \n" \
        "-par_pack [%i] (0/1: pack elements contigously for MPI_Isend in parallel) \n" \
        "-no_pack [%i] (0/1: skip packing remote elements to assess performance penalty) \n" \
        "-equilibrate [%i] (0/1: normalize rows of matrix) \n" \
        "--------------------------- Adaptive Precision Options --------------------------- \n" \
        "-ap_threshold_1 [%f] (float: threshold for two-way matrix partitioning for adaptive precision `-ap`) \n" \
        "-ap_threshold_2 [%f] (float: threshold for three-way matrix partitioning for adaptive precision `-ap`) \n" \
        "-dropout[%f] (0/1: enable dropout of elements below theh designated threshold) \n" \
        "-dropout_threshold [%f] (float: remove matrix elements below this range) \n\n",
        argv[0], \
        config->block_vec_size, \
        config->chunk_size, \
        config->sigma, \
        config->n_repetitions, \
        config->random_init_x, \
        value_type->c_str(), \
        seg_method->c_str(), \
        config->validate_result, \
        config->verbose_validation, \
        config->mode, \
        config->bench_time, \
        config->ba_synch, \
        config->comm_halos, \
        config->par_pack, \
        config->no_pack, \
        config->equilibrate, \
        config->ap_threshold_1, \
        config->ap_threshold_2, \
        config->dropout, \
        config->dropout_threshold);
                    
}


/**
    @brief Scan user cli input to variables, and verify that the entered parameters are valid
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *value_type : either single precision (/float/-sp) or double precision (/double/-dp)
    @param *random_init_x : decides if our generated x-vector is randomly generated, 
        or made from the default value defined in the DefaultValues struct
    @param *config : struct to initialze default values and user input
*/
void parse_cli_inputs(
    int argc,
    char *argv[],
    std::string *seg_method,
    std::string *kernel_format,
    std::string *value_type,
    Config *config,
    int my_rank)
{
    if (argc < 3)
    {
        if(my_rank == 0){cli_options_messge(argc, argv, seg_method, value_type, config);exit(1);}
    }

    config->matrix_file_name = argv[1];
    *kernel_format = argv[2];

    int args_start_index = 3;
    for (int i = args_start_index; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-c")
        {
            config->chunk_size = atoi(argv[++i]);

            if (config->chunk_size < 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-s")
        {

            config->sigma = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->sigma < 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-block_vec_size")
        {

            config->block_vec_size = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->block_vec_size < 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: number of x vectors must be > 1.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-bench_time" || arg == "-bench-time")
        {

            config->bench_time = atof(argv[++i]); // i.e. grab the NEXT

            if (config->bench_time < 0)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: bench_time must be > 0.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-rev")
        {
            config->n_repetitions = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->n_repetitions < 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: revisions must be >= 1.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-verbose")
        {
            config->verbose_validation = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->verbose_validation != 0 && config->verbose_validation != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: Only validation verbosity levels 0 and 1 are supported.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-validate")
        {
            config->validate_result = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->validate_result != 0 && config->validate_result != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to validate result (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-mode")
        {
            config->mode = *argv[++i]; // i.e. grab the NEXT

            if (config->mode != 'b' && config->mode != 's')
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: Only bench (b) and solve (s) modes are supported.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-rand_x" || arg == "-rand-x")
        {
            // config->random_init_x = atoi(argv[++i]); // i.e. grab the NEXT
            config->random_init_x = *(argv[++i]); // i.e. grab the NEXT

            // NOTE: were taking a char here, to allow for the case where x is taken as the mean
            if (config->random_init_x != '0' && config->random_init_x != '1' && config->random_init_x != 'm')
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to initialize x randomly (1, i.e. yes), fix x to 1 (0, i.e. no), or fix to the mean of matrix data (m, i.e. for mean).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-comm_halos" || arg == "-comm-halos")
        {
            config->comm_halos = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->comm_halos != 0 && config->comm_halos != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to communicate halo elements (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-ba_synch" || arg == "-ba-synch")
        {
            config->ba_synch = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->ba_synch != 0 && config->ba_synch != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to synchronize each iteration at barriers (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-par_pack" || arg == "-par-pack")
        {
            config->par_pack = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->par_pack != 0 && config->par_pack != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to pack contiguous elements for MPI_Isend in parallel (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-no_pack" || arg == "-no-pack")
        {
            config->no_pack = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->no_pack != 0 && config->no_pack != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to pack contiguous elements for MPI_Isend (1, i.e. yes) or skip (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-ap_threshold_1" || arg == "-apt1")
        {
            config->ap_threshold_1 = atof(argv[++i]); // i.e. grab the NEXT

            if (config->ap_threshold_1 < 0)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: ap_threshold_1 must be nonnegative.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-ap_threshold_2" || arg == "-apt2")
        {
            config->ap_threshold_2 = atof(argv[++i]); // i.e. grab the NEXT

            if (config->ap_threshold_2 < 0)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: ap_threshold_2 must be nonnegative.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-dropout" || arg == "-do")
        {
            config->dropout = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->dropout != 0 && config->dropout != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to droupout elements during partitioning (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-dropout_threshold" || arg == "-dt")
        {
            config->dropout_threshold = atof(argv[++i]); // i.e. grab the NEXT

            if (config->dropout_threshold < 0)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: dropout_threshold must be nonnegative.\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-equilibrate")
        {
            config->equilibrate = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->equilibrate != 0 && config->equilibrate != 1)
            {
                if(my_rank == 0){
                    fprintf(stderr, "ERROR: You can only choose to scale matrix elements by the max row and column element (1, i.e. yes) or not (0, i.e. no).\n");
                    cli_options_messge(argc, argv, seg_method, value_type, config);
                    exit(1);
                }
            }
        }
        else if (arg == "-dp")
        {
            *value_type = "dp";
        }
        else if (arg == "-sp")
        {
            *value_type = "sp";
        }
        else if (arg == "-hp")
        {
            *value_type = "hp";
        }
        else if (arg == "-ap[dp_sp]")
        {
            *value_type = "ap[dp_sp]";
        }
        else if (arg == "-ap[sp_hp]")
        {
            *value_type = "ap[sp_hp]";
        }
        else if (arg == "-ap[dp_hp]")
        {
            *value_type = "ap[dp_hp]";
        }
        else if (arg == "-ap[dp_sp_hp]")
        {
            *value_type = "ap[dp_sp_hp]";
        }
        else if (arg == "-seg_rows" || arg == "-seg-rows")
        {
            *seg_method = "seg-rows";
        }
        else if (arg == "-seg_nnz" || arg == "-seg-nnz")
        {
            *seg_method = "seg-nnz";
        }
        else if (arg == "-seg_metis" || arg == "-seg-metis")
        {
            *seg_method = "seg-metis";
        }
        else
        {
            if(my_rank == 0){
                fprintf(stderr, "ERROR: unknown argument: ");
                std::cout << arg << std::endl;
                cli_options_messge(argc, argv, seg_method, value_type, config);
                exit(1);
            }
        }
    }

    // Sanity checks //
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
    if (config->block_vec_size == 1){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Row-wise block vector layout selected, but block vector width is 1.\n "
            "Please choose colwise block vector layout if using SpMV.\n");
            exit(1);
        }
    }
#endif

    if (config->block_vec_size > 1){
        if(
            *value_type == "ap[dp_sp]" || 
            *value_type == "ap[dp_hp]" || 
            *value_type == "ap[sp_hp]" || 
            *value_type == "ap[dp_sp_hp]"){
            if(my_rank == 0){
                fprintf(stderr, "ERROR: SpMMV is not yet implemented for AP kernels.\n");
                exit(1);
            }
        }
    }

#ifdef __CUDACC__
    if (config->block_vec_size > 1){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: SpMMV is not yet implemented on GPUs.\n");
            exit(1);
        }
    }
#endif

#ifndef USE_MKL
    if (config->mode == 's'){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Solve mode (-mode s) selected, but USE_MKL not defined in Makefile.\n");
            exit(1);
        }
    }
#endif

#ifndef USE_METIS
    if (*seg_method == "seg-metis"){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: seg-metis selected, but USE_METIS not defined in Makefile.\n");
            exit(1);
        }
    }
#endif

#ifndef USE_MPI
    printf("USE_MPI not defined, forcing comm_halos = 0.\n");
    config->comm_halos = 0;
#endif

#ifdef USE_CUSPARSE
    if(*kernel_format != "crs" && *kernel_format != "csr" && *kernel_format != "scs"){
        if(config->sigma != 1){
            if(my_rank == 0){
                fprintf(stderr, "ERROR: At the moment CUSPARSE is only able to use the CRS format.\n");
                exit(1);
            }
        }
    }

    if(*value_type == "ap[dp_sp]" || *value_type == "ap[sp_hp]" || *value_type == "ap[dp_hp]" || *value_type == "ap[dp_sp_hp]"){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: cuSPARSE with Adaptive precision is not supported at this time.\n");
            exit(1);
        }
    }
#endif

#ifdef USE_MPI
    if(*value_type == "ap[dp_sp]" || *value_type == "ap[sp_hp]" || *value_type == "ap[dp_hp]" || *value_type == "ap[dp_sp_hp]"){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Adaptive precision with MPI is not supported at this time.\n");
            exit(1);
        }
    }
#ifdef HAVE_HALF_MATH
        // TODO!
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Half precision is not supported at the moment.\n");
            exit(1);
        }

        if(my_rank == 0){
            fprintf(stderr, "ERROR: Half precision is not supported when using MPI at the moment.\n");
            exit(1);
        }
#endif

// TODO!
#ifdef GRAPHTOPO_MPI_MODE
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Graph-based MPI communication not yet supported.\n");
            exit(1);
        }
#endif

#ifdef NO_MPI_MODE
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Please select an MPI communication mode (MPI_MODE).\n");
            exit(1);
        }
#endif
#endif

#ifndef HAVE_HALF_MATH
    if(*value_type == "hp" || *value_type == "ap[sp_hp]" || *value_type == "ap[dp_hp]" || *value_type == "ap[dp_sp_hp]"){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: Half precision selected, but HAVE_HALF_MATH not defined.\n");
            exit(1);
        }
    }
#endif

    if(*value_type != "ap[dp_sp]" && *value_type != "ap[sp_hp]" && *value_type != "ap[dp_hp]" && *value_type != "ap[dp_sp_hp]"){
        if(config->ap_threshold_1 > 0.0){
            fprintf(stderr, "WARNING: First adaptive precision threshold entered, but not used.\n");
        }
    }

    if(*value_type != "ap[dp_sp_hp]"){
        if(config->ap_threshold_2 > 0.0){
            fprintf(stderr, "WARNING: Second adaptive precision threshold entered, but three-way partitioning is not used.\n");
        }
    }

    if(*value_type == "ap[dp_sp]" || *value_type == "ap[sp_hp]" || *value_type == "ap[dp_hp]"){
        if(config->ap_threshold_1 == 0.0){
            fprintf(stderr, "WARNING: Two-way adaptive precision used, but the first threshold is not entered.\n");
        }
    }
    if(*value_type == "ap[dp_sp_hp]"){
        if(config->ap_threshold_1 == 0.0){
            fprintf(stderr, "WARNING: Three-way adaptive precision used, but the first threshold is not entered.\n");
        }
        if(config->ap_threshold_2 == 0.0){
            fprintf(stderr, "WARNING: Three-way adaptive precision used, but the second threshold is not entered.\n");
        }
        if(config->ap_threshold_1 <= config->ap_threshold_2){
            fprintf(stderr, "ERROR: Three-way adaptive precision used, but the second threshold is larger than the first.\n");
            exit(1);
        }
    }

    if(config->dropout){
        if(config->dropout_threshold == 0.0){
            fprintf(stderr, "WARNING: Dropout selected, but dropout_threshold is 0.\n");
        }
    }

    // Is this even true?
    // if (config->sigma > config->chunk_size)
    // {
    //     if(my_rank == 0){
    //         fprintf(stderr, "ERROR: sigma must be smaller than chunk size.\n");
    //         if(my_rank == 0){cli_options_messge(argc, argv, seg_method, value_type, config);exit(1);}
    //     }
    // }
    std::vector<std::string> acceptable_kernels{"crs", "csr", "scs"};
    if (std::find(std::begin(acceptable_kernels), std::end(acceptable_kernels), *kernel_format) == std::end(acceptable_kernels)){
        if(my_rank == 0){
            fprintf(stderr, "ERROR: kernel format not recognized.\n");
            exit(1);
        }
    }

    // if((*value_type == "ap" && *kernel_format != "crs") || (*value_type == "ap" && *kernel_format != "crs")){
    //     if(my_rank == 0){fprintf(stderr, "ERROR: only CRS kernel supports mixed precision at this time.\n");exit(1);}
    // }
}

#ifdef USE_SCAMAC
/* helper function:
 * split integer range [a...b-1] in n nearly equally sized pieces [ia...ib-1], for i=0,...,n-1 */
void split_range(ScamacIdx a, ScamacIdx b, ScamacIdx n, ScamacIdx i, ScamacIdx *ia, ScamacIdx *ib) {
  ScamacIdx m = (b-a-1)/n + 1;
  ScamacIdx d = n-(n*m -(b-a));
  if (i < d) {
    *ia = m*i + a;
    *ib = m*(i+1) + a;
  } else {
    *ia = m*d + (i-d)*(m-1) + a;
    *ib = m*d + (i-d+1)*(m-1) + a;
  }
}

// ScamacArgs parse_scamac(
//     int my_rank,
//     int comm_size
// ){

// }

// template <typename T> inline void sortPerm(T *arr, int *perm, int range_lo, int range_hi, bool rev=false)
// {
//     if(rev == false) {
//         std::stable_sort(perm+range_lo, perm+range_hi, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
//     } else {
//         std::stable_sort(perm+range_lo, perm+range_hi, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
//     }
// }
// #ifdef SCAMAC_H
/* our MPI error handler */
void my_mpi_error_handler() {
  fflush(stdout); // get the message out
  MPI_Abort(MPI_COMM_WORLD, 1);
}


void scamac_generate(
    Config *config,
    int my_rank,
    int comm_size,
    int* scamac_nrows,
    int* scamac_nnz,
    MtxData<double, int> *mtx
){

/**  examples/MPI/ex_count_mpi.c
 *
 *   basic example:
 *   - read a matrix name/argument string from the command line
 *   - count the number of non-zeros, and compute the maximum norm (=max |entry|) and row-sum norm
 *
 *   Matrix rows are generated in parallel MPI processes.
 *   The ScamacGenerator and ScamacWorkspace is allocated per process.
 */
//   int mpi_world_size, mpi_rank;
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  const char *matargstr = config->matrix_file_name.c_str();
//   if (argc<=1) {
//     printf("usage: ex_count <matrix-argument string>\n\nexample: ex_count Hubbard\n         ex_count Hubbard,n_sites=14,n_fermions=8,U=1.3\n         ex_count TridiagonalReal,subdiag=0.5,supdiag=2\n");
//     my_mpi_error_handler();
//   }
//   matargstr=argv[1];

  ScamacErrorCode err;
  ScamacGenerator *my_gen;
  char *errstr = NULL;
    
  // set error handler for MPI (the only global ScaMaC variable!)
  scamac_error_handler = my_mpi_error_handler;

  /* parse matrix name & parameters from command line to obtain a ScamacGenerator ... */
  /* an identical generator is created per MPI process */
  err = scamac_parse_argstr(matargstr, &my_gen, &errstr);
  /* ... and check for errors */
  if (err) {
    printf("-- Problem with matrix-argument string:\n-- %s\n---> Abort.\n",errstr);
    my_mpi_error_handler();
  }
  
  /* check matrix parameters */
  err = scamac_generator_check(my_gen, &errstr);
  if (err) {
    printf("-- Problem with matrix parameters:\n-- %s---> Abort.\n",errstr);
    my_mpi_error_handler();
  }
  
  /* finalize the generator ... */
  err=scamac_generator_finalize(my_gen);
  /* ... and check, whether the matrix dimension is too large */
  if (err==SCAMAC_EOVERFLOW) {
    // TODO: doesn't work with llvm
    // printf("-- matrix dimension exceeds max. IDX value (%"SCAMACPRIDX")\n---> Abort.\n",SCAMAC_IDX_MAX);
    my_mpi_error_handler();
  }
  /* catch remaining errors */
  SCAMAC_CHKERR(err);
  
  /* query number of rows and max. number of non-zero entries per row */
  ScamacIdx nrow = scamac_generator_query_nrow(my_gen);
  ScamacIdx maxnzrow = scamac_generator_query_maxnzrow(my_gen);

//   double t1 = MPI_Wtime();

  /* ScamacWorkspace is allocated per MPI process */
  ScamacWorkspace * my_ws;
  SCAMAC_TRY(scamac_workspace_alloc(my_gen, &my_ws));

  /* allocate memory for column indices and values per MPI process*/
//   ScamacIdx *cind = malloc(maxnzrow * sizeof(long int));
  ScamacIdx *cind = new signed long int[maxnzrow];
  double *val;
  if (scamac_generator_query_valtype(my_gen) == SCAMAC_VAL_REAL) {
    // val = malloc(maxnzrow * sizeof *val);
    val = new double[maxnzrow];
  } else {
    /* valtype == SCAMAC_VAL_COMPLEX */
    // val = malloc(2*maxnzrow * sizeof(double));
    val = new double[maxnzrow];
  }

  ScamacIdx ia,ib;
  // this MPI process generates rows ia ... ib-1
  split_range(0,nrow, comm_size, my_rank, &ia, &ib);
  
  // allocate space
  int* scamac_rowPtr = new int[nrow + 1];
  int* scamac_col = new int[maxnzrow * nrow];
  double* scamac_val = new double[maxnzrow * nrow];

  // init counters
  int row_ptr_idx = 0;
  int scs_arr_idx = 0;
  scamac_rowPtr[0] = 0;

  for (ScamacIdx idx=ia; idx<ib; idx++) {
    ScamacIdx k;
    /* generate single row ... */
    SCAMAC_TRY(scamac_generate_row(my_gen, my_ws, idx, SCAMAC_DEFAULT, &k, cind, val));
    /* ... which has 0 <=k <= maxnzrow entries */

    // Assign SCAMAC arrays to scs array
    scamac_rowPtr[row_ptr_idx + 1] = scamac_rowPtr[row_ptr_idx] + k;
    for(int i = 0; i < k; ++i){
        scamac_col[scs_arr_idx] = cind[i]; // I dont know if these are "remade" every iteration, seems like it
        scamac_val[scs_arr_idx] = val[i];
        ++scs_arr_idx;
    }

    *scamac_nnz += k;
    ++row_ptr_idx;
  }
  *scamac_nrows = ib - ia;

        // Stupid to convert back to COO, only to convert back to scs. But safe for now.
    (mtx->I).resize(*scamac_nnz);
    (mtx->J).resize(*scamac_nnz);
    (mtx->values).resize(*scamac_nnz); 

    // for (int i = 0; i < *scamac_nrows + 1; ++i){
    //     std::cout << "scamac row ptr[" << i << "] = " << scamac_rowPtr[i] << std::endl;
    // }

    int elem_num = 0;
    for(int row = 0; row < *scamac_nrows; ++row){
        for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
            (mtx->I)[elem_num] = row;
            (mtx->J)[elem_num] = scamac_col[idx];
            (mtx->values)[elem_num] = scamac_val[idx];
            ++elem_num;
        }
    }

    // Verify everything is working as expected
    // for (int i = 0; i < *scamac_nrows + 1; ++i){
    //     std::cout << "scamac row ptr[" << i << "] = " << scamac_rowPtr[i] << std::endl;
    // }
    // for(int row = 0; row < *scamac_nrows; ++row){
    //     for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
    //         std::cout << "row = " << row << std::endl;
    //         std::cout << "scamac_col[" << idx << "] = " << scamac_col[idx] << std::endl;
    //         std::cout << "scamac_val[" << idx << "] = " << scamac_val[idx] << std::endl;
    //     }
    // }
        // for(int idx = 0; idx < (mtx->I).size(); ++idx){
        //     // for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
        //         std::cout << "row[" << idx << "] = " << (mtx->I)[idx] << std::endl;
        //         std::cout << "col[" << idx << "] = " << (mtx->J)[idx] << std::endl;
        //         std::cout << "val[" << idx << "] = " << (mtx->values)[idx] << std::endl;
        //     }

  
  
  /* free local objects */
    delete[] scamac_rowPtr;
    delete[] scamac_col;
    delete[] scamac_val;
  free(cind);
  free(val);
  SCAMAC_TRY(scamac_workspace_free(my_ws));
  SCAMAC_TRY(scamac_generator_destroy(my_gen));
}
#endif

template <typename IT>
void generate_inv_perm(
    int *perm,
    int *inv_perm,
    int perm_len
){
    for(int i = 0; i < perm_len; ++i){
        // std::cout << "perm[" << i << "] = " << perm[i] << " <? " << perm_len << " = " << (perm[i] < perm_len) << std::endl;
        inv_perm[perm[i]] = i;
        // std::cout << "inv_perm[" << i << "] = " << inv_perm[perm[i]] << " <? " << perm_len << " = " << (inv_perm[perm[i]] < perm_len) << std::endl;
    }
}

template <typename VT, typename IT>
void apply_permutation(
    VT *permuted_vec,
    VT *vec_to_permute,
    IT *perm,
    int num_elems_to_permute
){
    #pragma omp parallel for
    for(int i = 0; i < num_elems_to_permute; ++i){
        permuted_vec[i] = vec_to_permute[perm[i]];
#ifdef DEBUG_MODE_FINE
        std::cout << "Permuting:" << static_cast<double>(vec_to_permute[i]) <<  " to " << static_cast<double>(vec_to_permute[perm[i]]) << std::endl;
#endif
    }
}

template <typename VT, typename IT>
void apply_strided_permutation(
    VT *permuted_vec,
    VT *vec_to_permute,
    IT *perm,
    int num_elems_to_permute,
    int stride
){
    #pragma omp parallel for
    for(int i = 0; i < num_elems_to_permute; ++i){
        permuted_vec[i*stride] = vec_to_permute[perm[i]*stride];
#ifdef DEBUG_MODE_FINE
        std::cout << "Permuting:" << vec_to_permute[i*stride] <<  " to " << vec_to_permute[perm[i]*stride] << std::endl;
#endif
    }
}

// TODO: hmm. Don't know if this is right
template<typename VT, typename IT>
void permute_scs_cols(
    ScsData<VT, IT> *scs,
    IT *perm
){
    ST n_scs_elements = scs->chunk_ptrs[scs->n_chunks - 1]
                    + scs->chunk_lengths[scs->n_chunks - 1] * scs->C;

    // std::vector<IT> col_idx_in_row(scs->n_rows_padded);

    std::vector<IT> col_perm_idxs(n_scs_elements);

    // TODO: parallelize

    for (ST i = 0; i < n_scs_elements; ++i) {
        if(scs->col_idxs[i] < scs->n_rows){
            // permuted version:
            col_perm_idxs[i] =  perm[scs->col_idxs[i]];
        }
        else{
            col_perm_idxs[i] = scs->col_idxs[i];
        }
    }

    // TODO (?): make col_perm_idx ptr, allocate on heap: parallelize
    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->col_idxs[i] = col_perm_idxs[i];
    }

}

template <typename T> inline void sortPerm(T *arr, int *perm, int range_lo, int range_hi, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+range_lo, perm+range_hi, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+range_lo, perm+range_hi, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

template <typename MT, typename VT, typename IT>
void convert_to_scs(
    MtxData<MT, IT> *local_mtx,
    ST C,
    ST sigma,
    ScsData<VT, IT> *scs,
    int *fixed_permutation = NULL,
    int *work_sharing_arr = nullptr,
    int my_rank = 0
    )
{
    scs->nnz    = local_mtx->nnz;
    scs->n_rows = local_mtx->n_rows;
    scs->n_cols = local_mtx->n_cols;

    scs->C = C;
    scs->sigma = sigma;

    if (scs->sigma % scs->C != 0 && scs->sigma != 1) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
        // }
#endif
    }

    if (will_add_overflow(scs->n_rows, scs->C)) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // }
    exit(1);
#endif        
        // return false;
    }
    scs->n_chunks      = (local_mtx->n_rows + scs->C - 1) / scs->C;

    if (will_mult_overflow(scs->n_chunks, scs->C)) {
#ifdef DEBUG_MODE
    // if(my_rank == 0){
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // }
    exit(1);
#endif   
        // return false;
    }
    scs->n_rows_padded = scs->n_chunks * scs->C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    // std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded);
    std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded + sigma);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        n_els_per_row[i].first = i;
    }

    for (ST i = 0; i < local_mtx->nnz; ++i) {
        ++n_els_per_row[local_mtx->I[i]].second;
    }

    // sort rows in the scope of sigma
    if (will_add_overflow(scs->n_rows_padded, scs->sigma)) {
        fprintf(stderr, "ERROR: no. of padded rows + sigma exceeds size type.\n");
        // return false;
    }

    if(fixed_permutation != NULL){
        std::vector<index_and_els_per_row> n_els_per_row_tmp(scs->n_rows_padded);
        for(int i = 0; i < scs->n_rows_padded; ++i){
            if(i < scs->n_rows){
                n_els_per_row_tmp[i].first = n_els_per_row[i].first;
                // n_els_per_row_tmp[i].second = n_els_per_row[fixed_permutation[i]].second;
                n_els_per_row_tmp[fixed_permutation[i]].second = n_els_per_row[i].second;
            }
            else{
                n_els_per_row_tmp[i].first = n_els_per_row[i].first;
                n_els_per_row_tmp[i].second = n_els_per_row[i].second;
            }
        }
        // n_els_per_row = n_els_per_row_tmp;
        for(int i = 0; i < scs->n_rows_padded; ++i){
            n_els_per_row[i] = n_els_per_row_tmp[i];
        }
    }
    else{
        for (ST i = 0; i < scs->n_rows_padded; i += scs->sigma) {
            auto begin = &n_els_per_row[i];
            auto end   = (i + scs->sigma) < scs->n_rows_padded
                            ? &n_els_per_row[i + scs->sigma]
                            : &n_els_per_row[scs->n_rows_padded];

            std::sort(begin, end,
                    // sort longer rows first
                    [](const auto & a, const auto & b) {
                        return a.second > b.second;
                    });
        }
    }

    scs->chunk_lengths = std::vector<IT>(scs->n_chunks + scs->sigma); // init a vector of length d.n_chunks
    scs->chunk_ptrs    = std::vector<IT>(scs->n_chunks + 1 + scs->sigma);

    IT cur_chunk_ptr = 0;
    
    for (ST i = 0; i < scs->n_chunks; ++i) {
        auto begin = &n_els_per_row[i * scs->C];
        auto end   = &n_els_per_row[i * scs->C + scs->C];

        scs->chunk_lengths[i] =
                std::max_element(begin, end,
                    [](const auto & a, const auto & b) {
                        return a.second < b.second;
                    })->second;

        if (will_add_overflow(cur_chunk_ptr, scs->chunk_lengths[i] * (IT)scs->C)) {
            fprintf(stderr, "ERROR: chunck_ptrs exceed index type.\n");
            // return false;
        }

        scs->chunk_ptrs[i] = cur_chunk_ptr;
        cur_chunk_ptr += scs->chunk_lengths[i] * scs->C;
    }

    ST n_scs_elements = scs->chunk_ptrs[scs->n_chunks - 1]
                        + scs->chunk_lengths[scs->n_chunks - 1] * scs->C;

    scs->chunk_ptrs[scs->n_chunks] = n_scs_elements;

    // construct permutation vector
    scs->old_to_new_idx = std::vector<IT>(scs->n_rows + scs->sigma);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < scs->n_rows) {
            scs->old_to_new_idx[old_row_idx] = i;
        }
    }
    
    scs->values   = std::vector<VT>(n_scs_elements + scs->sigma);
    scs->col_idxs = std::vector<IT>(n_scs_elements + scs->sigma);

#ifdef DEBUG_MODE
    printf("n_scs_elements = %i.\n\n", n_scs_elements);
#endif

    IT padded_col_idx = 0;

    // NOTE: I don't remember why I commented this out...
    // if(work_sharing_arr != nullptr){
    //     padded_col_idx = work_sharing_arr[my_rank];
    // }

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = VT{};
        // scs->col_idxs[i] = IT{};
        scs->col_idxs[i] = padded_col_idx;
    }

    // I don't know what this would help, but you can try it.
    // std::vector<IT> col_idx_in_row(scs->n_rows_padded);
    std::vector<IT> col_idx_in_row(scs->n_rows_padded + scs->sigma);
    // int *col_idx_in_row = new int [scs->n_rows_padded + scs->sigma];
    // for (int i = 0; i < scs->n_rows_padded + scs->sigma; ++i){
    //     col_idx_in_row[i] = 0;
    // }

    // fill values and col_idxs
    for (ST i = 0; i < scs->nnz; ++i) {
        IT row_old = local_mtx->I[i];
        IT row;

        if(fixed_permutation != NULL){
            row = fixed_permutation[row_old];
        }
        else{
            row = scs->old_to_new_idx[row_old];
        }

        ST chunk_index = row / scs->C;

        IT chunk_start = scs->chunk_ptrs[chunk_index];
        IT chunk_row   = row % scs->C;

        IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;

        scs->col_idxs[idx] = local_mtx->J[i];
        // TODO: Do you convert values correctly?
        scs->values[idx]   = local_mtx->values[i];

        col_idx_in_row[row]++;
    }

    // for (int i = 0; i < scs->n_rows_padded + scs->sigma; ++i){
    //     printf("col idx = %i\n", scs->col_idxs[i]);
    // }
    // exit(0);

    // printf("Problem row 16\n");
    // for (int i = 0; i < n_els_per_row[16].second; ++i){
    //     IT row = 16;
    //     ST chunk_index = row / scs->C;
    //     IT chunk_start = scs->chunk_ptrs[chunk_index];
    //     IT chunk_row   = row % scs->C;
    //     IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;
    //     printf("val: %f, col: %i\n", scs->values[idx], scs->col_idxs[idx]);
    // }

    // Sort inverse permutation vector, based on scs->old_to_new_idx
    // std::vector<int> inv_perm(scs->n_rows);
    // std::vector<int> inv_perm_temp(scs->n_rows);
    // std::iota(std::begin(inv_perm_temp), std::end(inv_perm_temp), 0); // Fill with 0, 1, ..., scs->n_rows.
    // generate_inv_perm<IT>(scs->old_to_new_idx.data(), &(inv_perm)[0],  scs->n_rows);


    int *inv_perm = new int[scs->n_rows + scs->sigma];
    int *inv_perm_temp = new int[scs->n_rows + scs->sigma];
    for(int i = 0; i < scs->n_rows; ++i){
        inv_perm_temp[i] = i;
    }
    // generate_inv_perm<IT>(scs->old_to_new_idx.data(), inv_perm, scs->n_rows + scs->sigma);
    generate_inv_perm<IT>(scs->old_to_new_idx.data(), inv_perm, scs->n_rows);


    scs->new_to_old_idx = inv_perm;

    scs->n_elements = n_scs_elements;

    // for(int i = 0; i < n_scs_elements; ++i){
    //     printf("col idx: %i\n", scs->col_idxs.data()[i]);
    // }

    // Experimental 2024_02_01, I do not want the rows permuted yet... so permute back
    // if sigma > C, I can see this being a problem
    // for (ST i = 0; i < scs->n_rows_padded; ++i) {
    //     IT old_row_idx = n_els_per_row[i].first;

    //     if (old_row_idx < scs->n_rows) {
    //         scs->old_to_new_idx[old_row_idx] = i;
    //     }
    // }    

    // return true;


    // printf("perm = [\n");
    // for (ST i = 0; i < scs->n_rows; ++i) {
    //     printf("perm idx %i\n", scs->old_to_new_idx.data()[i]);
    // }
    // printf("]\n");

    // printf("inv perm = [\n");
    // for (ST i = 0; i < scs->n_rows; ++i) {
    //     printf("inv perm idx %i\n", scs->new_to_old_idx[i]);
    // }
    // printf("]\n");
    // exit(1);

    // delete col_idx_in_row; <- uhh why does this cause a seg fault?
}

template<typename IT>
std::vector<IT> find_items1(std::vector<IT> const &v, int target) {
    std::vector<int> indices;
    auto it = v.begin();
    while ((it = std::find_if(it, v.end(), [&] (IT const &e) { return e == target; }))
        != v.end())
    {
        indices.push_back(std::distance(v.begin(), it));
        it++;
    }
    return indices;
}

template<typename IT>
std::vector<IT> find_items(std::vector<IT> const &v, int target) {
    std::vector<IT> indices;
 
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == target) {
            indices.push_back(i);
        }
    }
 
    return indices;
}


int cantor_pairing(int a, int b) {
    int c = .5 * (a + b) * (a + b + 1) + b;
    return c;
}
 

inline void sort_perm(int *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

void read_mtx(
    Config config,
    MtxData<double, int> *total_mtx,
    int my_rank)
{
    char* filename = const_cast<char*>(config.matrix_file_name.c_str());
    int nrows, ncols, nnz;

    MM_typecode matcode;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL) {printf("Unable to open file");}

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", filename);
        // return -1;
    }

    fclose(f);

    // bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode)||mm_is_integer(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool symm_flag = mm_is_symmetric(matcode);
    bool pattern_flag = mm_is_pattern(matcode);

    if(!compatible_flag)
    {
        printf("The matrix market file provided is not supported.\n Reason :\n");
        if(!mm_is_sparse(matcode))
        {
            printf(" * matrix has to be sparse\n");
        }

        if(!mm_is_real(matcode) && !(mm_is_pattern(matcode)))
        {
            printf(" * matrix has to be real or pattern\n");
        }

        if(!mm_is_symmetric(matcode) && !mm_is_general(matcode))
        {
            printf(" * matrix has to be either general or symmetric\n");
        }

        exit(0);
    }

    //int ncols;
    int *row_unsorted;
    int *col_unsorted;
    double *val_unsorted; // <- always read as double, then convert

    if(mm_read_unsymmetric_sparse<double, int>(filename, &nrows, &ncols, &nnz, &val_unsorted, &row_unsorted, &col_unsorted) < 0)
    {
        printf("Error in file reading\n");
        exit(1);
    }
    if(nrows != ncols)
    {
        printf("Matrix not square. Currently only square matrices are supported\n");
        exit(1);
    }

    //If matrix market file is symmetric; create a general one out of it
    if(symm_flag)
    {
        // printf("Creating a general matrix out of a symmetric one\n");

        int ctr = 0;

        //this is needed since diagonals might be missing in some cases
        for(int idx=0; idx<nnz; ++idx)
        {
            ++ctr;
            if(row_unsorted[idx]!=col_unsorted[idx])
            {
                ++ctr;
            }
        }

        int new_nnz = ctr;

        int *row_general = new int[new_nnz];
        int *col_general = new int[new_nnz];
        double *val_general = new double[new_nnz];

        int idx_gen=0;

        for(int idx=0; idx<nnz; ++idx)
        {
            row_general[idx_gen] = row_unsorted[idx];
            col_general[idx_gen] = col_unsorted[idx];
            val_general[idx_gen] = val_unsorted[idx];
            ++idx_gen;

            if(row_unsorted[idx] != col_unsorted[idx])
            {
                row_general[idx_gen] = col_unsorted[idx];
                col_general[idx_gen] = row_unsorted[idx];
                val_general[idx_gen] = val_unsorted[idx];
                ++idx_gen;
            }
        }

        free(row_unsorted);
        free(col_unsorted);
        free(val_unsorted);

        nnz = new_nnz;

        //assign right pointers for further proccesing
        row_unsorted = row_general;
        col_unsorted = col_general;
        val_unsorted = val_general;

        // delete[] row_general;
        // delete[] col_general;
        // delete[] val_general;
    }

    //permute the col and val according to row
    int* perm = new int[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row_unsorted, perm, nnz);

    int *col = new int[nnz];
    int *row = new int[nnz];
    double *val = new double[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        col[idx] = col_unsorted[perm[idx]];
        val[idx] = val_unsorted[perm[idx]];
        row[idx] = row_unsorted[perm[idx]];
    }

    delete[] perm;
    delete[] col_unsorted;
    delete[] val_unsorted;
    delete[] row_unsorted;

    total_mtx->values = std::vector<double>(val, val + nnz);
    total_mtx->I = std::vector<int>(row, row + nnz);
    total_mtx->J = std::vector<int>(col, col + nnz);
    total_mtx->n_rows = nrows;
    total_mtx->n_cols = ncols;
    total_mtx->nnz = nnz;
    total_mtx->is_sorted = 1; // TODO: not sure
    total_mtx->is_symmetric = 0; // TODO: not sure

    delete[] val;
    delete[] row;
    delete[] col;
}

template <typename VT, typename IT>
class SimpleDenseMatrix {
    public:
        std::vector<VT> vec;
        const ContextData<int> *local_context;
        int padding;

        SimpleDenseMatrix(void){
        }

        SimpleDenseMatrix(const ContextData<IT> *local_context, Config *config){
            // TODO: not too sure about this
            IT padding_from_heri = 0;

#ifdef USE_MPI
            padding_from_heri = (local_context->recv_counts_cumsum).back();   
#endif
            int padding = std::max(local_context->scs_padding, padding_from_heri);

            // For SCS, Each x column needs to be padded
            vec.resize(config->block_vec_size * (padding + local_context->num_local_rows), VT{});
        }

        // SimpleDenseMatrix(std::vector<VT> vec_to_copy, const ContextData<VT, IT> *local_context){
        //     // TODO: not too sure about this
        //     IT padding_from_heri = (local_context->recv_counts_cumsum).back();
        //     IT needed_padding = std::max(local_context->scs_padding, padding_from_heri);

        //     vec.resize(needed_padding + local_context->num_local_rows, 0);
        //     vec(vec_to_copy.begin(), vec_to_copy.end());
        // }

        void init(
            char vec_type,
            Config *config,
            int n_rows,
            int padding
            ){
            DefaultValues<VT, IT> default_values;

            if (config->random_init_x == 'm'){
                default_values.x = config->matrix_mean;
            }

            if (vec_type == 'x'){
            init_std_vec_with_ptr_or_value(
                config,
                vec, 
                vec.size(),
                n_rows,
                n_rows + padding,
                default_values.x,
                config->random_init_x);
            }
            else if (vec_type == 'y'){
            init_std_vec_with_ptr_or_value(
                config,
                vec, 
                vec.size(),
                n_rows,
                n_rows + padding,
                default_values.y,
                config->random_init_x);   
            }
        }



        // void populte(std::vector<double> vec_to_copy, const ContextData<IT> *local_context){
        //     IT padding_from_heri = (local_context->recv_counts_cumsum).back();
        //     IT needed_padding = std::max(local_context->scs_padding, padding_from_heri);

        //     vec.resize(needed_padding + local_context->num_local_rows, 0);
        //     vec(vec_to_copy.begin(), vec_to_copy.end());
        // }
};

template<typename VT>
class DenseMatrix {
public:

  DenseMatrix() {}

  DenseMatrix(int nr_, int nc_, int pad_) {
    resize(nr_, nc_, pad_);
  }

  void resize(int nr_, int nc_, int pad_) {
    nr = nr_;
    nc = nc_;
    pad = pad_;
    nr_padded = nr_ + pad_;
    vec.resize(nr_padded * nc);
  }

  VT* data() {
    return vec.data();
  }

  virtual void swap(DenseMatrix &other) {
    std::swap(nr, other.nr);
    std::swap(nc, other.nc);
    std::swap(nr_padded, other.nr_padded);
    std::swap(pad, other.pad);    
    
    vec.swap(other.vec);
  }

  void init(
    char vec_type,
    Config *config,
    int n_rows,
    int padding
    ){
#ifdef HAVE_HALF_MATH
        printf("Cannot cast into _Float16 at the moment.\n");
        exit(1);
#else
        DefaultValues<VT, int> default_values;


        if (config->random_init_x == 'm'){
            default_values.x = config->matrix_mean;
        }

        if (vec_type == 'x'){
        init_std_vec_with_ptr_or_value(
            config,
            vec, 
            vec.size(),
            n_rows,
            n_rows + padding,
            default_values.x,
            config->random_init_x);
        }
        else if (vec_type == 'y'){
        init_std_vec_with_ptr_or_value(
            config,
            vec, 
            vec.size(),
            n_rows,
            n_rows + padding,
            default_values.y,
            config->random_init_x);   
        }
#endif
    }
  
    virtual VT& operator()(int i, int j) = 0;

    int nr, nc;
    int pad;
    int nr_padded;
    std::vector<VT> vec;
};

template<typename VT>
class DenseMatrixRowMaj : public DenseMatrix<VT> {
public:
  DenseMatrixRowMaj(int nr_, int nc_, int pad_) : DenseMatrix<VT>(nr_, nc_, pad_) {}

  virtual VT& operator()(int i, int j) {
    return this->vec[j + this->nc * i];
  }

  int leading_dim() {
    return this->nc;
  }
  int row_stride() {
    return this->nc;
  }
};

template<typename VT>
class DenseMatrixColMaj : public DenseMatrix<VT> {
public:
  DenseMatrixColMaj(int nr_, int nc_, int pad_) : DenseMatrix<VT>(nr_, nc_, pad_) {}
 
  virtual VT& operator()(int i, int j) {
    return this->vec[i + this->nr_padded * j];
  }

  int leading_dim() {
    return this->nr_padded;
  }
  int row_stride() {
    return 1;
  }
};

template <typename VT, typename IT>
void extract_matrix_min_mean_max( 
    MtxData<VT, IT> *total_mtx,
    Config *config,
    int my_rank
){
    if(my_rank == 0){
    // Get max
        double max_val = 0;

        #pragma omp parallel for reduction(max:max_val) 
        for (int idx = 0; idx < total_mtx->nnz; idx++)
        max_val = max_val > fabs(static_cast<double>(total_mtx->values[idx])) ? max_val : fabs(static_cast<double>(total_mtx->values[idx]));

        //    max_val = max_val > fabs(total_mtx->values[idx]) ? max_val : fabs(total_mtx->values[idx]);

        // Get min
        double min_val = DBL_MAX;

        #pragma omp parallel for reduction(min:min_val) 
        for (int idx = 0; idx < total_mtx->nnz; idx++)
        min_val = min_val < fabs(static_cast<double>(total_mtx->values[idx])) ? min_val : fabs(static_cast<double>(total_mtx->values[idx]));

        // Take average and save max and min
        config->matrix_mean = min_val + ((max_val - min_val) / 2.0);
        config->matrix_max = max_val;
        config->matrix_min = min_val;

    #ifdef DEBUG_MODE
        printf("matrix_min = %f\n", config->matrix_min);
        printf("matrix_mean = %f\n", config->matrix_mean);
        printf("matrix_max = %f\n", config->matrix_max);
#endif
    }
#ifdef USE_MPI
    MPI_Bcast(&(config->matrix_min), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(config->matrix_mean), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(config->matrix_max), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
};

// template <typename VT, typename IT>
// void extract_largest_elems(
//     const MtxData<VT,IT> *coo_mat,
//     std::vector<VT> *largest_elems
// ){
//     // #pragma omp parallel for schedule (static)
//     // for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
//     //     if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
//     //         (*diag)[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
//     //     }
//     // }

//     #pragma omp parallel for schedule (static)
//     for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
//         int row = coo_mat->I[nz_idx];
//         // VT absValue = std::abs(coo_mat->values[nz_idx]);
//         double absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

//         #pragma omp critical
//         {
//             if (absValue > (*largest_elems)[row]) {
//                 (*largest_elems)[row] = absValue;
//             }
//         }
//     }
// };

// void extract_hp_diagonal(
//     const MtxData<double,int> *coo_mat,
//     std::vector<double> *diag
// ){
//     #pragma omp parallel for schedule (static)
//     for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
//         if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
//             (*diag)[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
//         }
//     }
// };

// void extract_lp_diagonal(
//     const MtxData<float,int> *coo_mat,
//     std::vector<double> *diag
// ){
//     #pragma omp parallel for schedule (static)
//     for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
//         if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
//             (*diag)[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
//         }
//     }
// };

// template <typename VT, typename IT>
// void scale_w_jacobi(
//     MtxData<VT,IT> *coo_mat,
//     std::vector<VT> *diag
// ){
//     #pragma omp parallel for schedule (static)
//     for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
//         coo_mat->values[nz_idx] = static_cast<VT>(coo_mat->values[nz_idx] / static_cast<double>((*diag)[coo_mat->I[nz_idx]]));
//     }

// };

template <typename VT, typename IT>
void extract_largest_row_elems(
    const MtxData<VT, IT> *coo_mat,
    std::vector<VT> *largest_row_elems
){
    // #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        int row = coo_mat->I[nz_idx];
        // VT absValue = std::abs(coo_mat->values[nz_idx]);
        VT absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

        // #pragma omp critical
        // {
            if (absValue > (*largest_row_elems)[row]) {
                (*largest_row_elems)[row] = absValue;
            // }
        }
    }
};

template <typename VT, typename IT>
void extract_largest_col_elems(
    const MtxData<VT, IT> *coo_mat,
    std::vector<VT> *largest_col_elems
){
    // #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        int col = coo_mat->J[nz_idx];
        // VT absValue = std::abs(coo_mat->values[nz_idx]);
        VT absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

        // #pragma omp critical
        // {
            if (absValue > (*largest_col_elems)[col]) {
                (*largest_col_elems)[col] = absValue;
            // }
        }
    }
};

template <typename VT, typename IT>
void scale_matrix_rows(
    MtxData<VT, IT> *coo_mat,
    std::vector<VT> *largest_row_elems
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        coo_mat->values[nz_idx] = coo_mat->values[nz_idx] / (*largest_row_elems)[coo_mat->I[nz_idx]];
    }
};

template <typename VT, typename IT>
void scale_matrix_cols(
    MtxData<VT, IT> *coo_mat,
    std::vector<VT> *largest_col_elems
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        coo_mat->values[nz_idx] = coo_mat->values[nz_idx] / (*largest_col_elems)[coo_mat->J[nz_idx]];
    }
};

template <typename VT, typename IT>
void equilibrate_matrix(MtxData<VT, IT> *coo_mat){
#ifdef HAVE_HALF_MATH
    std::vector<VT> largest_row_elems(coo_mat->n_cols, 0.0f16);
#else
    std::vector<VT> largest_row_elems(coo_mat->n_cols, 0.0);
#endif
    extract_largest_row_elems<VT, IT>(coo_mat, &largest_row_elems);
    scale_matrix_rows<VT, IT>(coo_mat, &largest_row_elems);

#ifdef HAVE_HALF_MATH
    std::vector<VT> largest_col_elems(coo_mat->n_cols, 0.0f16);
#else
    std::vector<VT> largest_col_elems(coo_mat->n_cols, 0.0);
#endif
    extract_largest_col_elems<VT, IT>(coo_mat, &largest_col_elems);
    scale_matrix_cols<VT, IT>(coo_mat, &largest_col_elems);
}

#ifdef USE_LIKWID
void register_likwid_markers(
    Config *config
){
    // Init parallel region
    #pragma omp parallel
    {
        if(config->kernel_format == "crs"){
            if(config->value_type == "ap[dp_sp]"){
                LIKWID_MARKER_REGISTER("spmv_apdpsp_crs_benchmark");
            }
            else if(config->value_type == "ap[dp_hp]"){
                LIKWID_MARKER_REGISTER("spmv_apdphp_crs_benchmark");
            }
            else if(config->value_type == "ap[sp_hp]"){
                LIKWID_MARKER_REGISTER("spmv_apsphp_crs_benchmark");
            }
            else if(config->value_type == "ap[dp_sp_hp]"){
                LIKWID_MARKER_REGISTER("spmv_apdpsphp_crs_benchmark");
            }
            else{
                if(config->block_vec_size > 1){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                    LIKWID_MARKER_REGISTER("block_colwise_spmv_crs_benchmark");  
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                    LIKWID_MARKER_REGISTER("block_rowwise_spmv_crs_benchmark");  
#endif   
                }
                else{
                    LIKWID_MARKER_REGISTER("spmv_crs_benchmark");  
                }
            }
        }
        else if(config->kernel_format == "scs"){
            if(
                config->chunk_size != 2 
                && config->chunk_size != 4
                && config->chunk_size != 8
                && config->chunk_size != 16
                && config->chunk_size != 32
                && config->chunk_size != 64
                && config->chunk_size != 128
                && config->chunk_size != 256
            ){
                // If we are not using an advanced version, then:
                if(config->value_type == "ap"){
                    LIKWID_MARKER_REGISTER("spmv_ap_scs_benchmark");
                }
                else{
                    if(config->block_vec_size > 1){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                        LIKWID_MARKER_REGISTER("block_colwise_spmv_scs_benchmark");  
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                        LIKWID_MARKER_REGISTER("block_rowwise_spmv_scs_benchmark");  
#endif   
                    }
                    else{
                        LIKWID_MARKER_REGISTER("spmv_scs_benchmark");
                    }
                }
            }
            else{
                // If we ARE using an advanced version, then:
                if(config->value_type == "ap"){
                    LIKWID_MARKER_REGISTER("spmv_ap_scs_adv_benchmark");
                }
                else{
                    if(config->block_vec_size > 1){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                        LIKWID_MARKER_REGISTER("block_colwise_spmv_scs_adv_benchmark");  
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                        LIKWID_MARKER_REGISTER("block_rowwise_spmv_scs_adv_benchmark");  
#endif   
                    }
                    else{
                        LIKWID_MARKER_REGISTER("spmv_scs_adv_benchmark");
                    }
                }
            }
        }
    }
}
#endif

void bogus_init_pin(int *num_devices, int my_rank){
    // Just to take overhead of pinning away from timers

#ifdef __CUDACC__
    CUDA_CHECK(cudaGetDeviceCount(num_devices));
    CUDA_CHECK(cudaSetDevice(my_rank));
    CUDA_CHECK(cudaDeviceSynchronize());
#else

    int num_threads;
    double bogus = 0.0;

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
#else
    num_threads = 1;
#endif
    #pragma omp parallel for
    for(int i = 0; i < num_threads; ++i){
        bogus += 1;
    }

    if(bogus < 100){
        printf("");
    }
#endif
}

/**
    @brief Matrix splitting routine, used for seperating a higher precision mtx coo struct into dp and sp sub-structs
    @param *local_mtx : Process-local coo matrix, received on this process from an earlier routine
    @param *dp_local_mtx : Process-local "higher precision" coo matrix
    @param *sp_local_mtx : Process-local "lower precision" coo matrix
*/
template <typename VT, typename IT>
void partition_precisions( 
    Config *config,
    MtxData<VT, IT> *local_mtx, // <- should be scaled when entering this routine 
    MtxData<double, int> *dp_local_mtx, 
    MtxData<float, int> *sp_local_mtx,
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *hp_local_mtx,
#endif
    std::vector<VT> *largest_row_elems, 
    std::vector<VT> *largest_col_elems,
    int my_rank = NULL)
{
    double threshold_1 = config->ap_threshold_1;
    double threshold_2 = config->ap_threshold_2;

    dp_local_mtx->is_sorted = local_mtx->is_sorted;
    dp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    dp_local_mtx->n_rows = local_mtx->n_rows;
    dp_local_mtx->n_cols = local_mtx->n_cols;

    sp_local_mtx->is_sorted = local_mtx->is_sorted;
    sp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    sp_local_mtx->n_rows = local_mtx->n_rows;
    sp_local_mtx->n_cols = local_mtx->n_cols;

#ifdef HAVE_HALF_MATH
    hp_local_mtx->is_sorted = local_mtx->is_sorted;
    hp_local_mtx->is_symmetric = local_mtx->is_symmetric;
    hp_local_mtx->n_rows = local_mtx->n_rows;
    hp_local_mtx->n_cols = local_mtx->n_cols;
#endif

    int dp_elem_ctr = 0;
    int sp_elem_ctr = 0;
#ifdef HAVE_HALF_MATH
    int hp_elem_ctr = 0;
#endif

    // TODO: This practice of assigning pointers to vectors is dangerous...
    std::vector<IT> dp_local_I;
    std::vector<IT> dp_local_J;
    std::vector<double> dp_local_vals;
    dp_local_mtx->I = dp_local_I;
    dp_local_mtx->J = dp_local_J;
    dp_local_mtx->values = dp_local_vals;

    std::vector<IT> sp_local_I;
    std::vector<IT> sp_local_J;
    std::vector<float> sp_local_vals;
    sp_local_mtx->I = sp_local_I;
    sp_local_mtx->J = sp_local_J;
    sp_local_mtx->values = sp_local_vals;

#ifdef HAVE_HALF_MATH
    std::vector<IT> hp_local_I;
    std::vector<IT> hp_local_J;
    std::vector<_Float16> hp_local_vals;
    hp_local_mtx->I = hp_local_I;
    hp_local_mtx->J = hp_local_J;
    hp_local_mtx->values = hp_local_vals;
#endif

    // Scan local_mtx
    // TODO: If this is a bottleneck:
    // 1. Scan in parallel 
    // 2. Allocate space
    // 3. Assign in parallel
    if(config->value_type == "ap[dp_sp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(config->equilibrate){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1 / \
                    ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else{
                    // else, place in sp_local_mtx 
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1){   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < threshold_1){
                    // else, place in sp_local_mtx 
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        sp_local_mtx->nnz = sp_elem_ctr;

        if(local_mtx->nnz != (dp_elem_ctr + sp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp and sp structs on rank: %i.\n", local_mtx->nnz - (dp_elem_ctr + sp_elem_ctr), my_rank);
            exit(1);
        }
    }
#ifdef HAVE_HALF_MATH
    else if(config->value_type == "ap[dp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(config->equilibrate){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else{
#ifdef HAVE_HALF_MATH
                    printf("Cannot cast into _Float16 at the moment.\n");
                    exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1){   
                    dp_local_mtx->values.push_back(static_cast<double>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < threshold_1){
#ifdef HAVE_HALF_MATH
                    printf("Cannot cast into _Float16 at the moment.\n");
                    exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;

        if(local_mtx->nnz != (dp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp and hp structs on rank: %i.\n", local_mtx->nnz - (dp_elem_ctr + hp_elem_ctr), my_rank);
            exit(1);
        }
    }
    else if(config->value_type == "ap[sp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(config->equilibrate){
                // TODO: static casting just to make it compile... 
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1 / \
                    ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) {   
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else{
                    // else, place in sp_local_mtx 
#ifdef HAVE_HALF_MATH
        printf("Cannot cast into _Float16 at the moment.\n");
        exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1){   
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else if (std::abs(static_cast<double>(local_mtx->values[i])) < threshold_1){
                    // else, place in sp_local_mtx 
#ifdef HAVE_HALF_MATH
                    printf("Cannot cast into _Float16 at the moment.\n");
                    exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
                else{
                    printf("partition_precisions ERROR: Element %i does not fit into either struct.\n", i);
                    exit(1);
                }
            }
        }

        sp_local_mtx->nnz = sp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;

        if(local_mtx->nnz != (sp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into sp and hp structs on rank: %i.\n", local_mtx->nnz - (sp_elem_ctr + hp_elem_ctr), my_rank);
            exit(1);
        }
    }
    else if(config->value_type == "ap[dp_sp_hp]"){
        for(int i = 0; i < local_mtx->nnz; ++i){
            // If element value below threshold, place in dp_local_mtx
            if(config->equilibrate){
                // Element is larger than the largest threshold
                if(
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1 / \
                    (static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]])))
                    ) {   
                    dp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if(
                    // Element is between thresholds
                    (std::abs(static_cast<double>(local_mtx->values[i])) <= threshold_1 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]]))) &&
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_2 / ( static_cast<double>((*largest_col_elems)[local_mtx->J[i]]) * static_cast<double>((*largest_row_elems)[local_mtx->I[i]])))
                ){
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else
                {
                    // else, element is between 0 and lowest threshold
#ifdef HAVE_HALF_MATH
        printf("Cannot cast into _Float16 at the moment.\n");
        exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
            else{
                // Element is larger than the largest threshold
                if(std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_1){
                    dp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    dp_local_mtx->I.push_back(local_mtx->I[i]);
                    dp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++dp_elem_ctr;
                }
                else if(
                    // Element is between thresholds
                    (std::abs(static_cast<double>(local_mtx->values[i])) <= threshold_1) &&
                    (std::abs(static_cast<double>(local_mtx->values[i])) >= threshold_2)
                ){
                    sp_local_mtx->values.push_back(static_cast<float>(local_mtx->values[i]));
                    sp_local_mtx->I.push_back(local_mtx->I[i]);
                    sp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++sp_elem_ctr;
                }
                else
                {
                    // else, element is between 0 and lowest threshold
#ifdef HAVE_HALF_MATH
                    printf("Cannot cast into _Float16 at the moment.\n");
                    exit(1);
                    // hp_local_mtx->values.push_back(static_cast<_Float16>(local_mtx->values[i]));
#endif
                    hp_local_mtx->I.push_back(local_mtx->I[i]);
                    hp_local_mtx->J.push_back(local_mtx->J[i]);
                    ++hp_elem_ctr;
                }
            }
        }

        dp_local_mtx->nnz = dp_elem_ctr;
        sp_local_mtx->nnz = sp_elem_ctr;
        hp_local_mtx->nnz = hp_elem_ctr;

        if(local_mtx->nnz != (dp_elem_ctr + sp_elem_ctr + hp_elem_ctr)){
            printf("partition_precisions ERROR: %i Elements have been lost when seperating \
            into dp, sp, and hp structs on rank: %i.\n", local_mtx->nnz - (dp_elem_ctr + sp_elem_ctr + hp_elem_ctr), my_rank);
            exit(1);
        }
    }
#endif
}

template <typename VT, typename IT>
void assign_spmv_kernel_cpu_data(
    Config *config,
    ContextData<IT> *local_context,
    int nzr_size,
    ScsData<VT, IT> *local_scs,
    VT **&to_send_elems,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    VT *local_y,
    double *dp_local_y,
    float *sp_local_y,
#ifdef HAVE_HALF_MATH
    _Float16 *hp_local_y,
#endif
    VT *local_x,
    VT *local_x_permuted,
    double *dp_local_x,
    double *dp_local_x_permuted,
    float *sp_local_x,
    float *sp_local_x_permuted,
#ifdef HAVE_HALF_MATH
    _Float16 *hp_local_x,
    _Float16 *hp_local_x_permuted,
#endif
    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded,
    MultiPrecKernelArgs<IT> *multi_prec_kernel_args_encoded
){
#ifdef DEBUG_MODE
     printf("Assigning data to cpu structs...\n");
#endif

#ifdef SINGLEVEC_MPI_MODE
    for(int i = 0; i < nzr_size; ++i){
        int nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size()];
    }
#endif
#ifdef MULTIVEC_MPI_MODE
// With multivec mode, we need to replicate the to_send buffers for each vector
for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
    for(int i = 0; i < nzr_size; ++i){
        int nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i + vec_idx * nzr_size] = new VT[local_context->comm_send_idxs[nz_recver].size()];
    }
}
#endif
#ifdef BULKVEC_MPI_MODE
// With bulkvec mode, the send buffers just need to be larger
    for(int i = 0; i < nzr_size; ++i){
        int nz_recver = local_context->non_zero_receivers[i];
        to_send_elems[i] = new VT[local_context->comm_send_idxs[nz_recver].size() * config->block_vec_size];
    }
#endif

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        multi_prec_kernel_args_encoded->dp_C = &dp_local_scs->C;
        multi_prec_kernel_args_encoded->dp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->dp_chunk_ptrs = dp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->dp_chunk_lengths = dp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->dp_col_idxs = dp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->dp_values = dp_local_scs->values.data();
        multi_prec_kernel_args_encoded->dp_local_x = dp_local_x_permuted;
        multi_prec_kernel_args_encoded->dp_local_y = dp_local_y;
        multi_prec_kernel_args_encoded->sp_C = &sp_local_scs->C;
        multi_prec_kernel_args_encoded->sp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->sp_chunk_ptrs = sp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->sp_chunk_lengths = sp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->sp_col_idxs = sp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->sp_values = sp_local_scs->values.data();
        multi_prec_kernel_args_encoded->sp_local_x = sp_local_x_permuted;
        multi_prec_kernel_args_encoded->sp_local_y = sp_local_y;
#ifdef HAVE_HALF_MATH
        multi_prec_kernel_args_encoded->hp_C = &hp_local_scs->C;
        multi_prec_kernel_args_encoded->hp_n_chunks = &dp_local_scs->n_chunks; //shared for now
        multi_prec_kernel_args_encoded->hp_chunk_ptrs = hp_local_scs->chunk_ptrs.data();
        multi_prec_kernel_args_encoded->hp_chunk_lengths = hp_local_scs->chunk_lengths.data();
        multi_prec_kernel_args_encoded->hp_col_idxs = hp_local_scs->col_idxs.data();
        multi_prec_kernel_args_encoded->hp_values = hp_local_scs->values.data();
        multi_prec_kernel_args_encoded->hp_local_x = hp_local_x_permuted;
        multi_prec_kernel_args_encoded->hp_local_y = hp_local_y;
#endif
    }
    else{
        // Encode kernel args into struct
        one_prec_kernel_args_encoded->C = &local_scs->C;
        one_prec_kernel_args_encoded->n_chunks = &local_scs->n_chunks;
        one_prec_kernel_args_encoded->chunk_ptrs = local_scs->chunk_ptrs.data();
        one_prec_kernel_args_encoded->chunk_lengths = local_scs->chunk_lengths.data();
        one_prec_kernel_args_encoded->col_idxs = local_scs->col_idxs.data();
        one_prec_kernel_args_encoded->values = local_scs->values.data();
        one_prec_kernel_args_encoded->local_x = local_x_permuted;
        one_prec_kernel_args_encoded->local_y = local_y;
    }
}

template <typename VT, typename IT>
void assign_mpi_args(
    CommArgs<VT, IT> *comm_args_encoded,
    ContextData<IT> *local_context,
#ifdef USE_MPI
    ScsData<VT, IT> *local_scs,
    const IT *work_sharing_arr,
    VT **to_send_elems,
    MPI_Request *recv_requests,
    MPI_Request *send_requests,
    int *nzs_size,
    int *nzr_size,
#ifdef __CUDACC__
    int *&d_perm, // device pointers for packing kernel
    int **&d_comm_send_idxs_ptr,
    int **&h_comm_send_idxs_ptr,
    int **&d_comm_send_idxs_data,
#endif
#endif
    int *my_rank,
    int *comm_size
){
#ifdef USE_MPI
#ifdef DEBUG_MODE
     printf("Assigning data to mpi structs...\n");
#endif
    MPI_Datatype MPI_ELEM_TYPE = get_mpi_type<VT>();
    // Encode comm args into struct
    comm_args_encoded->work_sharing_arr     = work_sharing_arr;
    comm_args_encoded->to_send_elems        = to_send_elems;
    comm_args_encoded->perm                 = local_scs->old_to_new_idx.data();
    comm_args_encoded->recv_requests        = recv_requests; // pointer to first element of array
    comm_args_encoded->nzs_size             = nzs_size;
    comm_args_encoded->send_requests        = send_requests;
    comm_args_encoded->nzr_size             = nzr_size;
    comm_args_encoded->num_local_rows       = &(local_context->num_local_rows);
    comm_args_encoded->per_vector_padding   = &(local_context->per_vector_padding);
    comm_args_encoded->MPI_ELEM_TYPE        = MPI_ELEM_TYPE;
#ifdef __CUDACC__
    CUDA_CHECK(cudaMalloc(&d_perm, local_scs->n_rows*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_perm, local_scs->old_to_new_idx.data(), local_scs->n_rows*sizeof(int), cudaMemcpyHostToDevice));
    comm_args_encoded->d_perm               = d_perm;

    // Allocate double pointer
    CUDA_CHECK(cudaMalloc(&d_comm_send_idxs_ptr, (*comm_size) * sizeof(int *)));

    // Allocate each individual pointer
    for(int i = 0; i < (*comm_size); ++i){
        int n_elem_to_send = local_context->comm_send_idxs[i].size();
        CUDA_CHECK(cudaMalloc((void **)&d_comm_send_idxs_data[i], n_elem_to_send*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_comm_send_idxs_data[i], local_context->comm_send_idxs[i].data(), n_elem_to_send*sizeof(int), cudaMemcpyHostToDevice));
        h_comm_send_idxs_ptr[i] = d_comm_send_idxs_data[i];
    }

    CUDA_CHECK(cudaMemcpy(d_comm_send_idxs_ptr, h_comm_send_idxs_ptr, (*comm_size) * sizeof(int *), cudaMemcpyHostToDevice));
    comm_args_encoded->d_comm_send_idxs     = d_comm_send_idxs_ptr;

#endif
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    comm_args_encoded->local_context        = local_context;
    comm_args_encoded->my_rank              = my_rank;
    comm_args_encoded->comm_size            = comm_size;
}

#ifdef __CUDACC__
template <typename VT, typename IT>
void assign_spmv_kernel_gpu_data(
    Config *config,
    ContextData<IT> *local_context,
    int nzr_size,
    ScsData<VT, IT> *local_scs,
#ifdef USE_MPI
    VT **&d_to_send_elems_ptr,
    VT **h_to_send_elems_ptr,
    VT **&d_to_send_elems_data,
#endif
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    VT *local_y,
    double *dp_local_y,
    float *sp_local_y,
#ifdef HAVE_HALF_MATH
    _Float16 *hp_local_y,
#endif
    VT *local_x,
    VT *local_x_permuted,
    double *dp_local_x,
    double *dp_local_x_permuted,
    float *sp_local_x,
    float *sp_local_x_permuted,
#ifdef HAVE_HALF_MATH
    _Float16 *hp_local_x,
    _Float16 *hp_local_x_permuted,
#endif
    VT *&d_x,
    VT *&d_y,
    ST *&d_C,
    ST *&d_n_chunks,
    IT *&d_chunk_ptrs,
    IT *&d_chunk_lengths,
    IT *&d_col_idxs,
    VT *&d_values,
    // ST *&d_n_blocks,
    double *&d_x_dp,
    double *&d_y_dp,
    ST *&d_C_dp,
    ST *&d_n_chunks_dp,
    IT *&d_chunk_ptrs_dp,
    IT *&d_chunk_lengths_dp,
    IT *&d_col_idxs_dp,
    double *d_values_dp,
    float *&d_x_sp,
    float *&d_y_sp,
    ST *&d_C_sp,
    ST *&d_n_chunks_sp,
    IT *&d_chunk_ptrs_sp,
    IT *&d_chunk_lengths_sp,
    IT *&d_col_idxs_sp,
    float *&d_values_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *&d_x_hp,
    _Float16 *&d_y_hp,
    ST *&d_C_hp,
    ST *&d_n_chunks_hp,
    IT *&d_chunk_ptrs_hp,
    IT *&d_chunk_lengths_hp,
    IT *&d_col_idxs_hp,
    _Float16  *&d_values_hp,
#endif
    ST n_blocks,
#ifdef USE_CUSPARSE
    CuSparseArgs *cusparse_args_encoded,
#endif
    OnePrecKernelArgs<VT, IT> *one_prec_kernel_args_encoded,
    MultiPrecKernelArgs<IT> *multi_prec_kernel_args_encoded
){
#ifdef DEBUG_MODE
    printf("Assigning data to gpu structs...\n");
#endif

#ifdef USE_CUSPARSE
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float     alpha           = 1.0f;
    float     beta            = 0.0f;

    CUDA_CHECK(cusparseCreate(&handle));

    if (config->kernel_format == "crs"){
        if(config->value_type == "dp"){
            CUDA_CHECK(cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F
            ));
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_64F));
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_64F));

            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else if(config->value_type == "sp"){
            CUDA_CHECK(cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_32F));
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_32F));

            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else if(config->value_type == "hp"){
            CUDA_CHECK(cusparseCreateCsr(&matA, local_scs->n_rows, local_scs->n_cols, local_scs->nnz, 
                d_chunk_ptrs, d_col_idxs, d_values,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_16F));
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_16F));

            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_16F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else{
            printf("CuSparse SpMV only enabled with CRS format in DP, SP, or HP\n");
            exit(1);
        }

    }
    else{
        if(config->value_type == "dp"){
            CUDA_CHECK(cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_64F
            ));
            // Create dense vector X
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_64F));
            // Create dense vector y
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_64F));
            // allocate an external buffer if needed
            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else if(config->value_type == "sp"){
            CUDA_CHECK(cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_32F
            ));
            // Create dense vector X
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_cols, d_x, CUDA_R_32F));
            // Create dense vector y
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows, d_y, CUDA_R_32F));
            // allocate an external buffer if needed
            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else if(config->value_type == "hp"){
            CUDA_CHECK(cusparseCreateSlicedEll(
                &matA, 
                local_scs->n_rows_padded, 
                local_scs->n_cols, 
                local_scs->nnz,
                local_scs->n_elements,
                local_scs->C,
                d_chunk_ptrs,
                d_col_idxs,
                d_values,
                CUSPARSE_INDEX_32I, 
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, 
                CUDA_R_16F
            ));
            // Create dense vector X
            CUDA_CHECK(cusparseCreateDnVec(&vecX, local_scs->n_rows_padded, d_x, CUDA_R_16F));
            // Create dense vector y
            CUDA_CHECK(cusparseCreateDnVec(&vecY, local_scs->n_rows_padded, d_y, CUDA_R_16F));
            // allocate an external buffer if needed
            CUDA_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY, CUDA_R_16F,
                CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
            ));
        }
        else{
            printf("CuSparse SELL-P only enabled with SCS format in DP, SP, or HP\n");
            exit(1);
        }
    }

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    cusparse_args_encoded->handle = handle;
    cusparse_args_encoded->opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparse_args_encoded->alpha = &alpha;
    cusparse_args_encoded->matA = matA;
    cusparse_args_encoded->vecX = vecX;
    cusparse_args_encoded->beta = &beta;
    cusparse_args_encoded->vecY = vecY;

    if(config->value_type == "dp"){
        cusparse_args_encoded->computeType = CUDA_R_64F;
    }
    else if(config->value_type == "sp"){
        cusparse_args_encoded->computeType = CUDA_R_32F;
    }
    else if(config->value_type == "hp"){
        cusparse_args_encoded->computeType = CUDA_R_16F;
    }

    cusparse_args_encoded->alg = CUSPARSE_SPMV_ALG_DEFAULT;
    cusparse_args_encoded->externalBuffer = dBuffer;
#else

    if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[sp_hp]" || config->value_type == "ap[dp_sp_hp]"){
        long n_scs_elements_dp = dp_local_scs->chunk_ptrs[dp_local_scs->n_chunks - 1]
                    + dp_local_scs->chunk_lengths[dp_local_scs->n_chunks - 1] * dp_local_scs->C;
        long n_scs_elements_sp = sp_local_scs->chunk_ptrs[sp_local_scs->n_chunks - 1]
                    + sp_local_scs->chunk_lengths[sp_local_scs->n_chunks - 1] * sp_local_scs->C;
#ifdef HAVE_HALF_MATH
        long n_scs_elements_hp = hp_local_scs->chunk_ptrs[hp_local_scs->n_chunks - 1]
                    + hp_local_scs->chunk_lengths[hp_local_scs->n_chunks - 1] * hp_local_scs->C;
#endif

        // TODO: temporary way to get around memory courruption problem
        for(int i = 0; i < n_scs_elements_dp; ++i){
            if(dp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad dp element %i found at idx %i\n", dp_local_scs->col_idxs[i], i);
#endif
                dp_local_scs->col_idxs[i] = 0;
            }
        }
        for(int i = 0; i < n_scs_elements_sp; ++i){
            if(sp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad sp %i element found at idx %i\n", sp_local_scs->col_idxs[i], i);
#endif
                sp_local_scs->col_idxs[i] = 0;
            }
        }
#ifdef HAVE_HALF_MATH
        for(int i = 0; i < n_scs_elements_hp; ++i){
            if(hp_local_scs->col_idxs[i] >= local_scs->n_rows){
#ifdef DEBUG_MODE
                printf("Bad sp %i element found at idx %i\n", hp_local_scs->col_idxs[i], i);
#endif
                hp_local_scs->col_idxs[i] = 0;
            }
        }
#endif


        // Allocate space for MP structs on device
        CUDA_CHECK(cudaMalloc(&d_values_dp, n_scs_elements_dp*sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_values_dp, &(dp_local_scs->values)[0], n_scs_elements_dp*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_values_sp, n_scs_elements_sp*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_values_sp, &(sp_local_scs->values)[0], n_scs_elements_sp*sizeof(float), cudaMemcpyHostToDevice));
#ifdef HAVE_HALF_MATH
        CUDA_CHECK(cudaMalloc(&d_values_hp, n_scs_elements_hp*sizeof(_Float16)));
        CUDA_CHECK(cudaMemcpy(d_values_hp, &(hp_local_scs->values)[0], n_scs_elements_hp*sizeof(_Float16), cudaMemcpyHostToDevice));
#endif

        CUDA_CHECK(cudaMalloc(&d_C_dp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_n_chunks_dp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_chunk_ptrs_dp, (dp_local_scs->n_chunks + 1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chunk_lengths_dp, dp_local_scs->n_chunks*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idxs_dp, n_scs_elements_dp*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C_sp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_n_chunks_sp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_chunk_ptrs_sp, (sp_local_scs->n_chunks + 1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chunk_lengths_sp, sp_local_scs->n_chunks*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idxs_sp, n_scs_elements_sp*sizeof(int)));
#ifdef HAVE_HALF_MATH
        CUDA_CHECK(cudaMalloc(&d_C_hp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_n_chunks_hp, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_chunk_ptrs_hp, (hp_local_scs->n_chunks + 1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chunk_lengths_hp, hp_local_scs->n_chunks*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idxs_hp, n_scs_elements_hp*sizeof(int)));
#endif

        // Copy matrix data to device
        CUDA_CHECK(cudaMemcpy(d_chunk_ptrs_dp, &(dp_local_scs->chunk_ptrs)[0], (dp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chunk_lengths_dp, &(dp_local_scs->chunk_lengths)[0], dp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idxs_dp, &(dp_local_scs->col_idxs)[0], n_scs_elements_dp*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_dp, &dp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_n_chunks_dp, &dp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_chunk_ptrs_sp, &(sp_local_scs->chunk_ptrs)[0], (sp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chunk_lengths_sp, &(sp_local_scs->chunk_lengths)[0], sp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idxs_sp, &(sp_local_scs->col_idxs)[0], n_scs_elements_sp*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_sp, &sp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_n_chunks_sp, &sp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice));

#ifdef HAVE_HALF_MATH
        CUDA_CHECK(cudaMemcpy(d_chunk_ptrs_hp, &(hp_local_scs->chunk_ptrs)[0], (hp_local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chunk_lengths_hp, &(hp_local_scs->chunk_lengths)[0], hp_local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idxs_hp, &(hp_local_scs->col_idxs)[0], n_scs_elements_hp*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C_hp, &hp_local_scs->C, sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_n_chunks_hp, &hp_local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice));
#endif

        // Copy x and y data to device
        double *local_x_dp_hardcopy = new double[local_scs->n_rows_padded];
        double *local_y_dp_hardcopy = new double[local_scs->n_rows_padded];
        float *local_x_sp_hardcopy = new float[local_scs->n_rows_padded];
        float *local_y_sp_hardcopy = new float[local_scs->n_rows_padded];
#ifdef HAVE_HALF_MATH
        _Float16 *local_x_hp_hardcopy = new _Float16[local_scs->n_rows_padded];
        _Float16 *local_y_hp_hardcopy = new _Float16[local_scs->n_rows_padded];
#endif

        #pragma omp parallel for
        for(int i = 0; i < local_scs->n_rows_padded; ++i){
            local_x_dp_hardcopy[i] = dp_local_x_permuted[i];
            local_y_dp_hardcopy[i] = dp_local_y[i];
            local_x_sp_hardcopy[i] = sp_local_x_permuted[i];
            local_y_sp_hardcopy[i] = sp_local_y[i];
#ifdef HAVE_HALF_MATH
            local_x_hp_hardcopy[i] = hp_local_x_permuted[i];
            local_y_hp_hardcopy[i] = hp_local_y[i];
#endif
        }

        CUDA_CHECK(cudaMalloc(&d_x_dp, local_scs->n_rows_padded*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_dp, local_scs->n_rows_padded*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x_sp, local_scs->n_rows_padded*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y_sp, local_scs->n_rows_padded*sizeof(float)));
#ifdef HAVE_HALF_MATH
        CUDA_CHECK(cudaMalloc(&d_x_sp, local_scs->n_rows_padded*sizeof(_Float16)));
        CUDA_CHECK(cudaMalloc(&d_y_sp, local_scs->n_rows_padded*sizeof(_Float16)));
#endif

        CUDA_CHECK(cudaMemcpy(d_x_dp, local_x_dp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_dp, local_y_dp_hardcopy, local_scs->n_rows_padded*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x_sp, local_x_sp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_sp, local_y_sp_hardcopy, local_scs->n_rows_padded*sizeof(float), cudaMemcpyHostToDevice));
#ifdef HAVE_HALF_MATH
        CUDA_CHECK(cudaMemcpy(d_x_hp, local_x_hp_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_hp, local_y_hp_hardcopy, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyHostToDevice));
#endif

        delete local_x_dp_hardcopy;
        delete local_y_dp_hardcopy;
        delete local_x_sp_hardcopy;
        delete local_y_sp_hardcopy;
#ifdef HAVE_HALF_MATH
        delete local_x_hp_hardcopy;
        delete local_y_hp_hardcopy;
#endif

        // Pack pointers into struct 
        // TODO: allow for each struct to have it's own C
        multi_prec_kernel_args_encoded->dp_C =             d_C_dp;
        multi_prec_kernel_args_encoded->dp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->dp_chunk_ptrs =    d_chunk_ptrs_dp;
        multi_prec_kernel_args_encoded->dp_chunk_lengths = d_chunk_lengths_dp;
        multi_prec_kernel_args_encoded->dp_col_idxs =      d_col_idxs_dp;
        multi_prec_kernel_args_encoded->dp_values =        d_values_dp;
        multi_prec_kernel_args_encoded->dp_local_x =       d_x_dp;
        multi_prec_kernel_args_encoded->dp_local_y =       d_y_dp;
        multi_prec_kernel_args_encoded->sp_C =             d_C_dp; // shared maybe don't?
        multi_prec_kernel_args_encoded->sp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->sp_chunk_ptrs =    d_chunk_ptrs_sp;
        multi_prec_kernel_args_encoded->sp_chunk_lengths = d_chunk_lengths_sp;
        multi_prec_kernel_args_encoded->sp_col_idxs =      d_col_idxs_sp;
        multi_prec_kernel_args_encoded->sp_values =        d_values_sp;
        multi_prec_kernel_args_encoded->sp_local_x =       d_x_sp;
        multi_prec_kernel_args_encoded->sp_local_y =       d_y_sp;
#ifdef HAVE_HALF_MATH
        multi_prec_kernel_args_encoded->hp_C =             d_C_dp; // shared maybe don't?
        multi_prec_kernel_args_encoded->hp_n_chunks =      d_n_chunks_dp; //shared for now
        multi_prec_kernel_args_encoded->hp_chunk_ptrs =    d_chunk_ptrs_hp;
        multi_prec_kernel_args_encoded->hp_chunk_lengths = d_chunk_lengths_hp;
        multi_prec_kernel_args_encoded->hp_col_idxs =      d_col_idxs_hp;
        multi_prec_kernel_args_encoded->hp_values =        d_values_hp;
        multi_prec_kernel_args_encoded->hp_local_x =       d_x_hp;
        multi_prec_kernel_args_encoded->hp_local_y =       d_y_hp;
#endif
        multi_prec_kernel_args_encoded->n_thread_blocks =         n_blocks;
    }
    else{
        long n_scs_elements = local_scs->chunk_ptrs[local_scs->n_chunks - 1]
                    + local_scs->chunk_lengths[local_scs->n_chunks - 1] * local_scs->C;

        if(config->value_type == "dp"){
            CUDA_CHECK(cudaMalloc(&d_values, n_scs_elements*sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(config->value_type == "sp"){
            CUDA_CHECK(cudaMalloc(&d_values, n_scs_elements*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(float), cudaMemcpyHostToDevice));
        }
        else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
            CUDA_CHECK(cudaMalloc(&d_values, n_scs_elements*sizeof(_Float16)));
            CUDA_CHECK(cudaMemcpy(d_values, &(local_scs->values)[0], n_scs_elements*sizeof(_Float16), cudaMemcpyHostToDevice));
#endif
        }
        
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_n_chunks, sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_chunk_ptrs, (local_scs->n_chunks + 1)*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_chunk_lengths, local_scs->n_chunks*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idxs, n_scs_elements*sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_chunk_ptrs, &(local_scs->chunk_ptrs)[0], (local_scs->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chunk_lengths, &(local_scs->chunk_lengths)[0], local_scs->n_chunks*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_idxs, &(local_scs->col_idxs)[0], n_scs_elements*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, &local_scs->C, sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_n_chunks, &local_scs->n_chunks, sizeof(long), cudaMemcpyHostToDevice));


#ifdef USE_MPI
        CUDA_CHECK(cudaMalloc(&d_to_send_elems_ptr, nzr_size * sizeof(VT *)));
#ifdef SINGLEVEC_MPI_MODE
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            CUDA_CHECK(cudaMalloc(&d_to_send_elems_data[i], local_context->comm_send_idxs[nz_recver].size() * sizeof(VT)));
            h_to_send_elems_ptr[i] = d_to_send_elems_data[i];
        }
#endif
#ifdef MULTIVEC_MPI_MODE
    // With multivec mode, we need to replicate the to_send buffers for each vector
    for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            CUDA_CHECK(cudaMalloc(&d_to_send_elems_data[i + vec_idx * nzr_size], local_context->comm_send_idxs[nz_recver].size() * sizeof(VT)));
            h_to_send_elems_ptr[i] = d_to_send_elems_data[i];
        }
    }
#endif
#ifdef BULKVEC_MPI_MODE
    // With bulkvec mode, the send buffers just need to be larger
        for(int i = 0; i < nzr_size; ++i){
            int nz_recver = local_context->non_zero_receivers[i];
            CUDA_CHECK(cudaMalloc(&d_to_send_elems_data[i], local_context->comm_send_idxs[nz_recver].size() * sizeof(VT)));
            h_to_send_elems_ptr[i] = d_to_send_elems_data[i];
        }
#endif
        CUDA_CHECK(cudaMemcpy(d_to_send_elems_ptr, h_to_send_elems_ptr, nzr_size * sizeof(VT *), cudaMemcpyHostToDevice));
#endif
        // Make type-specific copy to send to device
        VT *local_x_hardcopy = new VT[local_scs->n_rows_padded];
        VT *local_y_hardcopy = new VT[local_scs->n_rows_padded];

        #pragma omp parallel for
        for(int i = 0; i < local_scs->n_rows_padded; ++i){
            local_x_hardcopy[i] = local_x_permuted[i];
            local_y_hardcopy[i] = local_y[i];
        }

        CUDA_CHECK(cudaMalloc(&d_x, local_scs->n_rows_padded*sizeof(VT)));
        CUDA_CHECK(cudaMalloc(&d_y, local_scs->n_rows_padded*sizeof(VT)));

        CUDA_CHECK(cudaMemcpy(d_x, local_x_hardcopy, local_scs->n_rows_padded*sizeof(VT), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, local_y_hardcopy, local_scs->n_rows_padded*sizeof(VT), cudaMemcpyHostToDevice));

        delete local_x_hardcopy;
        delete local_y_hardcopy;

        // All args for kernel reside on the device
        one_prec_kernel_args_encoded->C =             d_C;
        one_prec_kernel_args_encoded->n_chunks =      d_n_chunks;
        one_prec_kernel_args_encoded->chunk_ptrs =    d_chunk_ptrs;
        one_prec_kernel_args_encoded->chunk_lengths = d_chunk_lengths;
        one_prec_kernel_args_encoded->col_idxs =      d_col_idxs;
        one_prec_kernel_args_encoded->values =        d_values;
        one_prec_kernel_args_encoded->local_x =       d_x;
        one_prec_kernel_args_encoded->local_y =       d_y;
        one_prec_kernel_args_encoded->n_thread_blocks =      n_blocks;
    }
#endif
}
#endif

template <typename VT, typename IT>
void copy_back_result(
    Config *config,
    ScsData<VT, IT> *local_scs,
    ScsData<double, IT> *dp_local_scs,
    ScsData<float, IT> *sp_local_scs,
#ifdef HAVE_HALF_MATH
    ScsData<_Float16, IT> *hp_local_scs,
#endif
    std::vector<VT> *local_y,
    VT * RESTRICT spmv_kernel_result,
    double * RESTRICT dp_spmv_kernel_result,
    float * RESTRICT sp_spmv_kernel_result,
#ifdef HAVE_HALF_MATH
    _Float16 * RESTRICT hp_spmv_kernel_result,
#endif
    double *dp_local_x_permuted,
    float *sp_local_x_permuted,
#ifdef HAVE_HALF_MATH
    _Float16 *hp_local_x_permuted,
#endif
    ContextData<IT> *local_context
){
#ifdef DEBUG_MODE
        printf("Entering copy_back_result\n");
#endif
        // Selects the first (n_rows)-many elements of a sorted y vector, and chops off padding
        std::vector<double> sorted_dp_local_y(local_y->size(), 0.0);
        std::vector<float> sorted_sp_local_y(local_y->size(), 0.0f);
#ifdef HAVE_HALF_MATH
        std::vector<_Float16> sorted_hp_local_y(local_y->size(), 0.0f16);
#endif

        // First, we have to copy the SpMV results back and un-permute (if applicable)
        if(config->value_type == "dp"){
#ifdef __CUDACC__
            // TODO: Integrate SpMMV on GPUs
            CUDA_CHECK(cudaMemcpy(dp_local_x_permuted, spmv_kernel_result, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost));
#else
            for(int i = 0; i < local_y->size(); ++i){
                dp_local_x_permuted[i] = spmv_kernel_result[i];
            }
#endif
            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                apply_permutation<double, IT>(&((sorted_dp_local_y.data())[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), &(dp_local_x_permuted[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), local_scs->old_to_new_idx.data(), local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                apply_strided_permutation<double, IT>(&((sorted_dp_local_y.data())[vec_idx]), &(dp_local_x_permuted[vec_idx]), local_scs->old_to_new_idx.data(), local_scs->n_rows, config->block_vec_size);
#endif
            }
        }
        else if(config->value_type == "sp"){
#ifdef __CUDACC__
            CUDA_CHECK(cudaMemcpy(sp_local_x_permuted, spmv_kernel_result, local_scs->n_rows_padded*sizeof(float), cudaMemcpyDeviceToHost));
#else
            for(int i = 0; i < local_y->size(); ++i){
                sp_local_x_permuted[i] = spmv_kernel_result[i];
            }
#endif
            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                apply_permutation<float, IT>(&((sorted_sp_local_y.data())[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), &(sp_local_x_permuted[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), local_scs->old_to_new_idx.data(), local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                apply_strided_permutation<float, IT>(&((sorted_sp_local_y.data())[vec_idx]), &(sp_local_x_permuted[vec_idx]), local_scs->old_to_new_idx.data(), local_scs->n_rows, config->block_vec_size);
#endif
            }
        }
        else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
#ifdef __CUDACC__
            CUDA_CHECK(cudaMemcpy(hp_local_x_permuted->data(), spmv_kernel_result, local_scs->n_rows_padded*sizeof(_Float16), cudaMemcpyDeviceToHost));

#else
            for(int i = 0; i < local_y->size(); ++i){
                hp_local_x_permuted[i] = spmv_kernel_result[i];
            }
#endif
            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                apply_permutation<_Float16, IT>(&((sorted_hp_local_y.data())[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), &(hp_local_x_permuted[vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]), local_scs->old_to_new_idx.data(), local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                apply_strided_permutation<_Float16, IT>(&((sorted_hp_local_y.data())[vec_idx]), &(hp_local_x_permuted[vec_idx]), local_scs->old_to_new_idx.data(), local_scs->n_rows, config->block_vec_size);
#endif
            }
#endif
        }
        else if(config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[dp_sp_hp]"){
            // TODO: Validate this!
#ifdef __CUDACC__
            CUDA_CHECK(cudaMemcpy(dp_local_x_permuted, dp_spmv_kernel_result, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost));
#else
            for(int i = 0; i < local_y->size(); ++i){
                dp_local_x_permuted[i] = dp_spmv_kernel_result[i];
            }
#endif
            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                apply_permutation<double, IT>(&((sorted_dp_local_y.data())[vec_idx * (dp_local_scs->n_rows + local_context->per_vector_padding)]), &(dp_local_x_permuted[vec_idx * (dp_local_scs->n_rows + local_context->per_vector_padding)]), dp_local_scs->old_to_new_idx.data(), dp_local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                apply_strided_permutation<double, IT>(&((sorted_dp_local_y.data())[vec_idx]), &(dp_local_x_permuted[vec_idx]), dp_local_scs->old_to_new_idx.data(), dp_local_scs->n_rows, config->block_vec_size);
#endif
            }
        }
        else if(config->value_type == "ap[sp_hp]"){
#ifdef HAVE_HALF_MATH
#ifdef __CUDACC__
            CUDA_CHECK(cudaMemcpy(sp_local_x_permuted, sp_spmv_kernel_result, local_scs->n_rows_padded*sizeof(double), cudaMemcpyDeviceToHost));
#else
            for(int i = 0; i < local_y->size(); ++i){
                sp_local_x_permuted[i] = sp_spmv_kernel_result[i];
            }
#endif
            for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
                apply_permutation<float, IT>(&((sorted_sp_local_y.data())[vec_idx * (sp_local_scs->n_rows + local_context->per_vector_padding)]), &(sp_local_x_permuted[vec_idx * (sp_local_scs->n_rows + local_context->per_vector_padding)]), sp_local_scs->old_to_new_idx.data(), sp_local_scs->n_rows);
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
                apply_strided_permutation<float, IT>(&((sorted_sp_local_y.data())[vec_idx]), &(sp_local_x_permuted[vec_idx]), sp_local_scs->old_to_new_idx.data(), sp_local_scs->n_rows, config->block_vec_size);
#endif
            }
#endif
        }
        // NOTE: With AP, we need to permute w.r.t. the local_scs of the highest precision

    // Then, give USpMV output to local_y for validation. This part will take block vector padding off if applicable

    // TODO: Bandaid
    // TODO: Allow to be row-wise for performance comparison against MKL
    // TODO: Ugly. No raw loops!
// Need to take care to skip padding elements, since this is what is padded to MKL
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
        for(int i = 0; i < local_scs->n_rows; ++i){
            for(int j = 0; j < config->block_vec_size; ++j){
                (*local_y)[i + local_scs->n_rows * j] = sorted_dp_local_y[i * config->block_vec_size + j];
            }
        }
#endif


#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
        int vec_idx = 0;
        int shift = 0;
        for(int i = 0; i < local_scs->n_rows * config->block_vec_size; ++i){
            if(vec_idx < local_scs->n_rows){
                if(config->value_type == "dp" || config->value_type == "ap[dp_sp]" || config->value_type == "ap[dp_hp]" || config->value_type == "ap[dp_sp_hp]"){
                    (*local_y)[i] = sorted_dp_local_y[i + shift];
#ifdef DEBUG_MODE_FINE
                    printf("Assigning %f from idx %i of sorted_local_y to local_y[%i]\n", sorted_dp_local_y[i + shift], i + shift, i);
#endif
                }
                else if(config->value_type == "sp" || config->value_type == "ap[sp_hp]"){
                    (*local_y)[i] = static_cast<double>(sorted_sp_local_y[i + shift]);
                }
                else if(config->value_type == "hp"){
#ifdef HAVE_HALF_MATH
                    (*local_y)[i] = static_cast<double>(sorted_hp_local_y[i + shift]);
#endif
                }
                
            }

            if(vec_idx == local_scs->n_rows - 1){
                vec_idx = 0;
                shift += local_context->per_vector_padding;
                continue;
            }

            ++vec_idx;
        } 
#endif

        // Manually resize for ease later on (and I don't see a better way)
        local_y->resize(local_context->num_local_rows * config->block_vec_size);
}

template <typename VT, typename IT>
void copy_data_to_ap_vectors(
    SimpleDenseMatrix<double, IT> *dp_local_x,
    SimpleDenseMatrix<float, IT> *sp_local_x,
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> *hp_local_x,
#endif
    VT *local_x_vec,
    long x_vec_size,
    SimpleDenseMatrix<double, IT> *dp_local_y,
    SimpleDenseMatrix<float, IT> *sp_local_y,
#ifdef HAVE_HALF_MATH
    SimpleDenseMatrix<_Float16, IT> *hp_local_y,
#endif
    VT *local_y_vec,
    long y_vec_size
){
    // Copy initialized RHS and LHS into other precisions
    for(int i = 0; i < x_vec_size; ++i){
        dp_local_x->vec[i] = static_cast<double>(local_x_vec[i]);
        sp_local_x->vec[i] = static_cast<float>(local_x_vec[i]);
#ifdef HAVE_HALF_MATH
        // hp_local_x->vec[i] = static_cast<_Float16>(local_x_vec[i]);
        printf("Cannot cast into _Float16 at the moment.\n");
        exit(1);
#endif
    }

    for(int i = 0; i < y_vec_size; ++i){
        dp_local_y->vec[i] = 0.0;
        sp_local_y->vec[i] = 0.0f;
#ifdef HAVE_HALF_MATH
        hp_local_y->vec[i] = 0.0f16;
#endif
    }
}

#endif
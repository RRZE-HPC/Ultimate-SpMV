#ifndef UTILITIES
#define UTILITIES

#include <mpi.h>
#include <cstdarg>
#include <random>
#include <iomanip>
#include <limits>
#include <algorithm>

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

template <typename VT, typename IT>
using V = Vector<VT, IT>;

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

template <typename VT, typename IT>
void print_vector(const std::string &name,
             const V<VT, IT> &v)
{
    print_vector(name, v.data(), v.data() + v.n_rows);
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

template <typename IT>
void
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
    std::vector<VT> y_ref(mtx.n_rows);

    ST nnz = mtx.nnz;
    if (mtx.I.size() != mtx.J.size() || mtx.I.size() != mtx.values.size())
    {
        fprintf(stderr, "ERROR: %s:%d sizes of rows, cols, and values differ.\n", __FILE__, __LINE__);
        exit(1);
    }

    for (ST i = 0; i < nnz; ++i)
    {
        y_ref[mtx.I[i]] += mtx.values[i] * x[mtx.J[i]];
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
void random_init(VT *begin, VT *end)
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    srand(time(NULL) + my_rank);

    // std::mt19937 engine;

    // if (!g_same_seed_for_every_vector)
    // {
    //     std::random_device rnd_device;
    //     engine.seed(rnd_device());
    // }

    std::uniform_real_distribution<double> dist(0.1, 2.0);

    for (VT *it = begin; it != end; ++it)
    {
        *it = ((VT) rand() / ((VT) RAND_MAX)) + 1;
    }
}

template <typename VT, typename IT>
void random_init(V<VT, IT> &v)
{
    random_init(v.data(), v.data() + v.n_rows);
}

template <typename VT, typename IT>
void init_with_ptr_or_value(V<VT, IT> &x,
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
void init_std_vec_with_ptr_or_value(
    std::vector<VT> &x,
    ST n_x,
    VT default_value,
    bool init_with_random_numbers = false,
    const std::vector<VT> *x_in = nullptr
)
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


/**
    @brief Scan user cli input to variables, and verify that the entered parameters are valid
    @param *file_name_str : name of the matrix-matket format data, taken from the cli
    @param *seg_method : the method by which the rows of mtx are partiitoned, either by rows or by number of non zeros
    @param *value_type : either single precision (/float/-sp) or double precision (/double/-dp)
    @param *random_init_x : decides if our generated x-vector is randomly generated, 
        or made from the default value defined in the DefaultValues struct
    @param *config : struct to initialze default values and user input
*/
void verify_and_assign_inputs(
    int argc,
    char *argv[],
    std::string *file_name_str,
    std::string *seg_method,
    std::string *value_type,
    Config *config)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                        "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
        exit(1);
    }

    *file_name_str = argv[1];

    int args_start_index = 2;
    for (int i = args_start_index; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-c")
        {
            config->chunk_size = atoi(argv[++i]);

            if (config->chunk_size < 1)
            {
                fprintf(stderr, "ERROR: chunk size must be >= 1.\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-s")
        {

            config->sigma = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->sigma < 1)
            {
                fprintf(stderr, "ERROR: sigma must be >= 1.\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-rev")
        {
            config->n_repetitions = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->n_repetitions < 1)
            {
                fprintf(stderr, "ERROR: revisions must be >= 1.\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-verbose")
        {
            config->verbose_validation = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->verbose_validation != 0 && config->verbose_validation != 1)
            {
                fprintf(stderr, "ERROR: Only validation verbosity levels 0 and 1 are supported.\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-validate")
        {
            config->validate_result = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->validate_result != 0 && config->validate_result != 1)
            {
                fprintf(stderr, "ERROR: You can only choose to validate result (1, i.e. yes) or not (0, i.e. no).\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-mode")
        {
            config->mode = *argv[++i]; // i.e. grab the NEXT

            if (config->mode != 'b' && config->mode != 's')
            {
                fprintf(stderr, "ERROR: Only bench (b) and solve (s) modes are supported.\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-rand_x")
        {
            config->random_init_x = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->random_init_x != 0 && config->random_init_x != 1)
            {
                fprintf(stderr, "ERROR: You can only choose to initialize x randomly (1, i.e. yes) or not (0, i.e. no).\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-log_prof")
        {
            config->log_prof = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->log_prof != 0 && config->log_prof != 1)
            {
                fprintf(stderr, "ERROR: You can only choose to activate log profiler (1, i.e. yes) or not (0, i.e. no).\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
            }
        }
        else if (arg == "-comm_halos")
        {
            config->comm_halos = atoi(argv[++i]); // i.e. grab the NEXT

            if (config->comm_halos != 0 && config->comm_halos != 1)
            {
                fprintf(stderr, "ERROR: You can only choose to communicate halo elements (1, i.e. yes) or not (0, i.e. no).\n");
                fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                                "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                        argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
                exit(1);
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
        else if (arg == "-seg_rows")
        {
            *seg_method = "seg-rows";
        }
        else if (arg == "-seg_nnz")
        {
            *seg_method = "seg-nnz";
        }
        else
        {
            fprintf(stderr, "ERROR: unknown argument.\n");
            fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                            "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                    argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
            exit(1);
        }
    }

    if (config->sigma > config->chunk_size)
    {
        fprintf(stderr, "ERROR: sigma must be smaller than chunk size.\n");
        fprintf(stderr, "Usage: %s martix-market-filename [options]\n"
                        "options [defaults]: -c [%li], -s [%li], -rev [%li], -rand_x [%i], -sp/dp [%s], -seg_nnz/seg_rows [%s], -validate [%i], -verbose [%i], -mode [%c], -log_prof [%i], -comm_halos [%i]\n",
                argv[0], config->chunk_size, config->sigma, config->n_repetitions, config->random_init_x, value_type->c_str(), seg_method->c_str(), config->validate_result, config->verbose_validation, config->mode, config->log_prof, config->comm_halos);
        exit(1);
    }
}

/**
    @brief Convert mtx struct to sell-c-sigma data structures.
    @param *local_mtx : process local mtx data structure, that was populated by  
    @param C : chunk height
    @param sigma : sorting scope
    @param *scs : The ScsData struct to populate with data
*/
template <typename VT, typename IT>
void convert_to_scs(
    const MtxData<VT, IT> *local_mtx,
    ST C,
    ST sigma,
    ScsData<VT, IT> *scs,
    int *work_sharing_arr = nullptr,
    int my_rank = 0)
{
    scs->nnz    = local_mtx->nnz;
    scs->n_rows = local_mtx->n_rows;
    scs->n_cols = local_mtx->n_cols;

    scs->C = C;
    scs->sigma = sigma;

    if (scs->sigma % scs->C != 0 && scs->sigma != 1) {
        fprintf(stderr, "NOTE: sigma is not a multiple of C\n");
    }

    if (will_add_overflow(scs->n_rows, scs->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // return false;
    }
    scs->n_chunks      = (local_mtx->n_rows + scs->C - 1) / scs->C;

    if (will_mult_overflow(scs->n_chunks, scs->C)) {
        fprintf(stderr, "ERROR: no. of padded row exceeds size type.\n");
        // return false;
    }
    scs->n_rows_padded = scs->n_chunks * scs->C;

    // first enty: original row index
    // second entry: population count of row
    using index_and_els_per_row = std::pair<ST, ST>;

    std::vector<index_and_els_per_row> n_els_per_row(scs->n_rows_padded);

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

    // determine chunk_ptrs and chunk_lengths

    // TODO: check chunk_ptrs can overflow
    // std::cout << d.n_chunks << std::endl;
    scs->chunk_lengths = V<IT, IT>(scs->n_chunks); // init a vector of length d.n_chunks
    scs->chunk_ptrs    = V<IT, IT>(scs->n_chunks + 1);

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

    scs->old_to_new_idx = V<IT, IT>(scs->n_rows);

    for (ST i = 0; i < scs->n_rows_padded; ++i) {
        IT old_row_idx = n_els_per_row[i].first;

        if (old_row_idx < scs->n_rows) {
            scs->old_to_new_idx[old_row_idx] = i;
        }
    }
    

    scs->values   = V<VT, IT>(n_scs_elements);
    scs->col_idxs = V<IT, IT>(n_scs_elements);

    IT padded_col_idx = 0;

    if(work_sharing_arr != nullptr){
        padded_col_idx = work_sharing_arr[my_rank];
    }

    for (ST i = 0; i < n_scs_elements; ++i) {
        scs->values[i]   = VT{};
        // scs->col_idxs[i] = IT{};
        scs->col_idxs[i] = padded_col_idx;
    }

    std::vector<IT> col_idx_in_row(scs->n_rows_padded);

    // fill values and col_idxs
    for (ST i = 0; i < scs->nnz; ++i) {
        IT row_old = local_mtx->I[i];

        IT row = scs->old_to_new_idx[row_old];

        ST chunk_index = row / scs->C;

        IT chunk_start = scs->chunk_ptrs[chunk_index];
        IT chunk_row   = row % scs->C;

        IT idx = chunk_start + col_idx_in_row[row] * scs->C + chunk_row;

        scs->col_idxs[idx] = local_mtx->J[i];
        scs->values[idx]   = local_mtx->values[i];

        col_idx_in_row[row]++;
    }

    scs->n_elements = n_scs_elements;

    // return true;
}

// A basic class, to make a vector from the mtx context
template <typename VT, typename IT>
class SimpleDenseMatrix {
    public:
        std::vector<VT> vec;

        SimpleDenseMatrix(const ContextData<VT, IT> *local_context){
            // TODO: not too sure about this
            IT padding_from_heri = (local_context->recv_counts_cumsum).back();
            IT needed_padding = std::max(local_context->scs_padding, padding_from_heri);

            vec.resize(needed_padding + local_context->amnt_local_elems, 0);
        }

        void init(Config *config){
            DefaultValues<VT, IT> default_values;
            init_std_vec_with_ptr_or_value(
                vec, 
                vec.size(),
                default_values.x, 
                config->random_init_x);
        }
};

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
 

#endif
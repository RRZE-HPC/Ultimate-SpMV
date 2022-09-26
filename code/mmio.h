/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#ifndef MM_IO_H
#define MM_IO_H

#include "classes_structs.hpp"

#include <stdio.h>
#include <algorithm>
#include <map>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I_, int **J, 
        double **val, MM_typecode *matcode);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode);		/* too complex for a macro */


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	((*typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)
#define mm_set_sparse(typecode)	mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)	((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)	((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
									(*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17


/******************** Matrix Market internal definitions ********************
   MM_matrix_typecode: 4-character sequence
				    ojbect 		sparse/   	data        storage 
						  		dense     	type        scheme
   string position:	 [0]        [1]			[2]         [3]
   Matrix typecode:  M(atrix)  C(oord)		R(eal)   	G(eneral)
						        A(array)	C(omplex)   H(ermitian)
											P(attern)   S(ymmetric)
								    		I(nteger)	K(kew)
 ***********************************************************************/

#define MM_MTX_STR		"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate" 
#define MM_SPARSE_STR	"coordinate"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR		"real"
#define MM_INT_STR		"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR		"symmetric"
#define MM_HERM_STR		"hermitian"
#define MM_SKEW_STR		"skew-symmetric"
#define MM_PATTERN_STR  "pattern"


/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int _I[], int J[],
		 double val[], MM_typecode matcode);
int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int _I[], int J[],
		double val[], MM_typecode matcode);
int mm_read_mtx_crd_entry(FILE *f, int *_I, int *J, double *real, double *img,
			MM_typecode matcode);

// NOTE: must be fully implemented here, as it is a templated function
template <typename VT, typename IT>
int mm_read_unsymmetric_sparse(
    const char *fname,
    IT *M_,
    IT *N_,
    IT *nz_,
    VT **val_,
    IT **I_,
    IT **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz; // TODO: leave as int for now, so we dont have to change mm_read_mtx_crd_size
    IT i;
    VT *val;
    IT *II, *J;

    if ((f = fopen(fname, "r")) == NULL)
            return -1;


    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }


    // if ( !((mm_is_real(matcode)||mm_is_pattern(matcode)) && mm_is_matrix(matcode) &&
    //         mm_is_sparse(matcode)))
    if ( !((mm_is_real(matcode)||mm_is_pattern(matcode)||mm_is_integer(matcode)) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }

    *M_ = M;
    *N_ = N;
    *nz_ = nz;

    /* reseve memory for matrices */

    II = (IT *) malloc(nz * sizeof(IT));
    J = (IT *) malloc(nz * sizeof(IT));
    val = (VT *) malloc(nz * sizeof(VT));

    *val_ = val;
    *I_ = II;
    *J_ = J;

    if(mm_is_pattern(matcode))
    {
        // printf("pattern matrix all non-zero values will be set to 0.01.\n");

        for(int i=0; i<nz; ++i)
        {
            val[i] = 0.01;
        }
    }

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    if(!mm_is_pattern(matcode))
    {
        // for (i=0; i<nz; i++)
        // {
        //     if (fscanf(f, "%d %d %lg\n", &II[i], &J[i], &val[i]) // NOTE: what does the lg affect?
        //             != 3) return MM_PREMATURE_EOF;
        //     II[i]--;  /* adjust from 1-based to 0-based */
        //     J[i]--;
        // }
        if(typeid(VT) == typeid(double)){
            for (i=0; i<nz; i++)
            {
                if (fscanf(f, "%d %d %lg\n", &II[i], &J[i], &val[i]) // NOTE: what does the lg affect?
                        != 3) return MM_PREMATURE_EOF;
                II[i]--;  /* adjust from 1-based to 0-based */
                J[i]--;
            }
        }
        else if(typeid(VT) == typeid(float)){
            for (i=0; i<nz; i++)
            {
                if (fscanf(f, "%d %d %f\n", &II[i], &J[i], &val[i]) // NOTE: what does the lg affect?
                        != 3) return MM_PREMATURE_EOF;
                II[i]--;  /* adjust from 1-based to 0-based */
                J[i]--;
            }
        }
    }
    else
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d\n", &II[i], &J[i])
                    != 2) return MM_PREMATURE_EOF;
            II[i]--;  /* adjust from 1-based to 0-based */
            J[i]--;
        }
    }

    fclose(f);

    return 0;
}

inline void sort_perm(int *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

template <typename VT, typename IT>
void read_mtx(
    const std::string matrix_file_name,
    Config config,
    MtxData<VT, IT> *total_mtx,
    int my_rank)
{
    char* filename = const_cast<char*>(matrix_file_name.c_str());
    IT nrows, ncols, nnz;
    VT *val_ptr;
    IT *I_ptr;
    IT *J_ptr;

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
    IT *row_unsorted;
    IT *col_unsorted;
    VT *val_unsorted;

    if(mm_read_unsymmetric_sparse<VT, IT>(filename, &nrows, &ncols, &nnz, &val_unsorted, &row_unsorted, &col_unsorted) < 0)
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

        IT *row_general = new IT[new_nnz];
        IT *col_general = new IT[new_nnz];
        VT *val_general = new VT[new_nnz];

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
    IT* perm = new IT[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row_unsorted, perm, nnz);

    IT *col = new IT[nnz];
    IT *row = new IT[nnz];
    VT *val = new VT[nnz];

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

    total_mtx->values = std::vector<VT>(val, val + nnz);
    total_mtx->I = std::vector<IT>(row, row + nnz);
    total_mtx->J = std::vector<IT>(col, col + nnz);
    total_mtx->n_rows = nrows;
    total_mtx->n_cols = ncols;
    total_mtx->nnz = nnz;
    total_mtx->is_sorted = 1; // TODO: not sure
    total_mtx->is_symmetric = 0; // TODO: not sure

    delete[] val;
    delete[] row;
    delete[] col;
}

void test_func(void);



#endif
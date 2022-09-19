/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#ifndef MM_IO_H
#define MM_IO_H

#include <stdio.h>

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

void test_func(void);



#endif
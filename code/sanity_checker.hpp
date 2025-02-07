#ifndef SANITY_CHECKER_HPP
#define SANITY_CHECKER_HPP

#include <stdio.h>
#include "classes_structs.hpp"
#include "utilities.hpp"
#include <vector>

class SanityChecker {
public:

	template<typename VT, typename IT>
	static void check_vectors_before_comm(
		Config *config,
		ScsData<VT, IT> *local_scs,
		ContextData<IT> *local_context,
		SpmvKernel<VT, IT> *spmv_kernel,
		int my_rank
	) {
		printf("Before comm spmv_kernel.local_x on rank %i = [\n", my_rank);
		// TODO: Integrate ROWWISE
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
		for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i){
				for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i * config->block_vec_size + vec_idx]);
		}
#endif
		printf("]\n");

		printf("Before comm spmv_kernel.local_y on rank %i = [\n", my_rank);
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_y[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
// #endif
		printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_vectors_after_comm(
		Config *config,
		ScsData<VT, IT> *local_scs,
		ContextData<IT> *local_context,
		SpmvKernel<VT, IT> *spmv_kernel,
		int my_rank
	) {
			printf("After comm spmv_kernel.local_x on rank %i = [\n", my_rank);
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
#endif
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
		for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i){
				for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i * config->block_vec_size + vec_idx]);
		}
#endif
		printf("]\n");

		printf("After comm spmv_kernel.local_y on rank %i = [\n", my_rank);
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_y[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
// #endif
		printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_vectors_after_spmv(
		Config *config,
		ScsData<VT, IT> *local_scs,
		ContextData<IT> *local_context,
		SpmvKernel<VT, IT> *spmv_kernel,
		int my_rank
	) {
			printf("spmv_kernel->local_x after spmv_kernel.execute() on rank %i = [\n", my_rank);
			// TODO: Integrate ROWWISE
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
							printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
			}
// #endif
			printf("]\n");

			printf("spmv_kernel->local_y after spmv_kernel.execute() on rank %i = [\n", my_rank);
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
							printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_y[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
			}
// #endif
			printf("]\n");

	};

	template<typename VT, typename IT>
	static void check_vectors_after_swap(
		Config *config,
		ScsData<VT, IT> *local_scs,
		ContextData<IT> *local_context,
		SpmvKernel<VT, IT> *spmv_kernel,
		int my_rank
	) {
			printf("After spmv_kernel.swap_local_vectors() local_x on rank %i = [\n", my_rank);
		// TODO: Integrate ROWWISE
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_x[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
// #endif
		printf("]\n");

		printf("After spmv_kernel.swap_local_vectors() local_y on rank %i = [\n", my_rank);
// #ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
				for(int i = 0; i < local_scs->n_rows + local_context->per_vector_padding; ++i)
						printf("vec %i: %f,\n", vec_idx, spmv_kernel->local_y[i + vec_idx * (local_scs->n_rows + local_context->per_vector_padding)]);
		}
// #endif
		printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_vectors_before_gather(
		Config *config,
		ContextData<IT> *local_context,
		Result<VT, IT> *r,
		int my_rank
	) {
				printf("Gathering results: rank %i local_x_mkl_copy = [\n", my_rank);
			// TODO: Integrate ROWWISE
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < local_context->num_local_rows; ++i)
							printf("vec %i: %f,\n", vec_idx, r->x_out[i + vec_idx * local_context->num_local_rows]);
			}
#endif
			printf("]\n");

			printf("Gathering results: rank %i local_y = [\n", my_rank);
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < local_context->num_local_rows; ++i)
							printf("vec %i: %f,\n", vec_idx, r->y_out[i + vec_idx * local_context->num_local_rows]);
			}
#endif
			printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_vectors_after_gather(
		Config *config,
		ContextData<IT> *local_context,
		Result<VT, IT> *r,
		int my_rank
	) {
			printf("global_x_mkl_copy = [\n");
			// TODO: Integrate ROWWISE
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < r->total_rows; ++i)
							printf("vec %i: %f,\n", vec_idx, r->total_x[i + vec_idx * r->total_rows]);
			}
#endif
			printf("]\n");

			printf("global_y = [\n");
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
			for(int vec_idx = 0; vec_idx < config->block_vec_size; ++vec_idx){
					for(int i = 0; i < r->total_rows; ++i)
							printf("vec %i: %f,\n", vec_idx, r->total_uspmv_result[i + vec_idx * r->total_rows]);
			}
#endif
			printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_perm_vectors(
		ScsData<VT, IT> *local_scs
	) {
    printf("perm = [");
    for(int i = 0; i < local_scs->n_rows; ++i){
        printf("%i, ", local_scs->old_to_new_idx[i]);
    }
    printf("]\n");

    printf("inv_perm = [");
    for(int i = 0; i < local_scs->n_rows; ++i){
        printf("%i, ", local_scs->new_to_old_idx[i]);
    }
    printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_vector_padding(
		Config *config,
		ScsData<VT, IT> local_scs,
		ContextData<IT> local_context,
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		DenseMatrixColMaj<VT> local_x
#else
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
		DenseMatrixRowMaj<VT> local_x
#else
		SimpleDenseMatrix<VT, IT> local_x
#endif
#endif
	){
		// check vectors are "padded" correctly
		printf("local_x (size %i)= [\n", local_x.vec.size());
		if(local_x.vec.size() != (local_scs.n_rows + local_context.per_vector_padding) * config->block_vec_size){
				printf("ERROR: local_x size not expected.\n");
				exit(0);
		}
				
		for(int i = 0; i < local_x.vec.size(); ++i){
				printf("%f, \n", local_x.vec[i]);
		}
		printf("]\n");
	};

	template<typename VT, typename IT>
	static void check_local_x_vectors(
		std::vector<VT> local_x_mkl_copy,
#ifdef COLWISE_BLOCK_VECTOR_LAYOUT
		DenseMatrixColMaj<VT> local_x
#else
#ifdef ROWWISE_BLOCK_VECTOR_LAYOUT
		DenseMatrixRowMaj<VT> local_x
#else
		SimpleDenseMatrix<VT, IT> local_x
#endif
#endif
	){
		printf("local_x = [\n");
		for(int i = 0; i < local_x.vec.size(); ++i){
				printf("%f, \n", local_x.vec[i]);
		}
		printf("]\n");

		printf("local_x_mkl_copy = [\n");
		for(int i = 0; i < local_x_mkl_copy.size(); ++i){
				printf("%f, \n", local_x_mkl_copy[i]);
		}
		printf("]\n");
	};


};
#endif

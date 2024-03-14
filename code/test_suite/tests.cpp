#include "catch.hpp"
#include "classes_structs.hpp"
#include "mpi_funcs.hpp"
#include "test_data/test_data.hpp"

int test_cntr = 0;

TEST_CASE("Validate COO matrix splitting function 'seperate_lp_from_hp' ", "[require]"){
    SECTION("Test splitting with COO Matrix M1"){
        std::cout << "Test " << ++test_cntr << ": M1 seperate_lp_from_hp..." << std::endl;

        // Declare structs to be tested
        MtxData<double, int> M1_hp;
        MtxData<float, int> M1_lp;

        // Perform function to test
        seperate_lp_from_hp<double, int>(1.0, &M1, &M1_hp, &M1_lp);
        M1_hp^exp_M1_hp; // <- check for discrepancies in matrix data
        REQUIRE((M1_hp == exp_M1_hp)); // <- If none, test will pass

        M1_lp^exp_M1_lp;
        REQUIRE((M1_lp == exp_M1_lp));
    }
}

TEST_CASE("Validate function 'convert_to_scs' for c = 1 and sigma = 1 fixed", "[require]"){
    SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){
        std::cout << "Test " << ++test_cntr << ": M1 convert_to_scs..." << std::endl;

        exp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_scs_1_1);

        ScsData<double, int> M1_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1, 1, 1, &M1_scs_1_1);

        M1_scs_1_1^exp_M1_scs_1_1;
        REQUIRE((M1_scs_1_1 == exp_M1_scs_1_1));
    }
    SECTION("Test scs conversion with split COO Matrix exp_M1_hp, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_hp convert_to_scs..." << std::endl;

        exp_M1_hp_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_hp_scs_1_1);

        ScsData<double, int> M1_hp_scs_1_1;

        convert_to_scs<double, int>(1.0, &exp_M1_hp, 1, 1, &M1_hp_scs_1_1);

        M1_hp_scs_1_1^exp_M1_hp_scs_1_1;
        REQUIRE((M1_hp_scs_1_1 == exp_M1_hp_scs_1_1));
    }
    SECTION("Test scs conversion with split COO Matrix exp_M1_lp, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_lp convert_to_scs..." << std::endl;

        exp_M1_lp_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_lp_scs_1_1);

        ScsData<float, int> M1_lp_scs_1_1;

        convert_to_scs<float, int>(1.0, &exp_M1_lp, 1, 1, &M1_lp_scs_1_1);

        M1_lp_scs_1_1^exp_M1_lp_scs_1_1;
        REQUIRE((M1_lp_scs_1_1 == exp_M1_lp_scs_1_1));
    }
    SECTION("Test scs conversion with top empty row, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_te convert_to_scs..." << std::endl;

        exp_M1_te_row_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_te_row_scs_1_1);

        ScsData<double, int> M1_te_row_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_te_row, 1, 1, &M1_te_row_scs_1_1);

        M1_te_row_scs_1_1^exp_M1_te_row_scs_1_1;
        REQUIRE((M1_te_row_scs_1_1 == exp_M1_te_row_scs_1_1));
    }
    SECTION("Test scs conversion with middle empty row, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_me convert_to_scs..." << std::endl;

        exp_M1_me_row_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_me_row_scs_1_1);

        ScsData<double, int> M1_me_row_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_me_row, 1, 1, &M1_me_row_scs_1_1);

        M1_me_row_scs_1_1^exp_M1_me_row_scs_1_1;
        REQUIRE((M1_me_row_scs_1_1 == exp_M1_me_row_scs_1_1));
    }
    SECTION("Test scs conversion with bottom empty row, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_be convert_to_scs..." << std::endl;

        exp_M1_be_row_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_be_row_scs_1_1);

        ScsData<double, int> M1_be_row_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_be_row, 1, 1, &M1_be_row_scs_1_1);

        M1_be_row_scs_1_1^exp_M1_be_row_scs_1_1;
        REQUIRE((M1_be_row_scs_1_1 == exp_M1_be_row_scs_1_1));
    }
    SECTION("Test scs conversion with left empty col, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_le convert_to_scs..." << std::endl;

        exp_M1_le_col_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_le_col_scs_1_1);

        ScsData<double, int> M1_le_col_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_le_col, 1, 1, &M1_le_col_scs_1_1);

        M1_le_col_scs_1_1^exp_M1_le_col_scs_1_1;
        REQUIRE((M1_le_col_scs_1_1 == exp_M1_le_col_scs_1_1));
    }
    SECTION("Test scs conversion with middle empty col, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_me convert_to_scs..." << std::endl;

        exp_M1_me_col_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_me_col_scs_1_1);

        ScsData<double, int> M1_me_col_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_me_col, 1, 1, &M1_me_col_scs_1_1);

        M1_me_col_scs_1_1^exp_M1_me_col_scs_1_1;
        REQUIRE((M1_me_col_scs_1_1 == exp_M1_me_col_scs_1_1));
    }
    SECTION("Test scs conversion with bottom empty row and right empty column, c=1, s=1"){
        std::cout << "Test " << ++test_cntr << ": M1_bere convert_to_scs..." << std::endl;

        exp_M1_be_row_re_col_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_be_row_re_col_scs_1_1);

        ScsData<double, int> M1_be_row_re_col_scs_1_1;

        convert_to_scs<double, int>(1.0, &M1_be_row_re_col, 1, 1, &M1_be_row_re_col_scs_1_1);

        M1_be_row_re_col_scs_1_1^exp_M1_be_row_re_col_scs_1_1;
        REQUIRE((M1_be_row_re_col_scs_1_1 == exp_M1_be_row_re_col_scs_1_1));
    }
}
// TEST_CASE("Validate function 'convert_to_scs' for c > 1 and sigma = 1 fixed", "[require]"){
//     SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){

//     }

TEST_CASE("Validate function 'convert_to_scs' for sigma > 1 and c = 1 fixed", "[require]"){
    SECTION("Test scs conversion with COO Matrix M_big, c=1, s=2"){
        std::cout << "Test " << ++test_cntr << ": M_big convert_to_scs_1_2 and compress..." << std::endl;

        MtxData<double, int> hp_M_big;
        MtxData<float, int> lp_M_big;

        // First, validate splitting routine
        seperate_lp_from_hp<double,int>(1.0, &M_big, &hp_M_big, &lp_M_big);

        lp_M_big^exp_M_big_lp;
        REQUIRE((lp_M_big == exp_M_big_lp));

        hp_M_big^exp_M_big_hp;
        REQUIRE((hp_M_big == exp_M_big_hp));

        // Then validate hp and lp COO structs convert sell_1_2 structs correctly
        exp_M_big_lp_scs_1_2.assign_explicit_test_data(&explicit_exp_M_big_lp_scs_1_2);
        exp_M_big_hp_scs_1_2.assign_explicit_test_data(&explicit_exp_M_big_hp_scs_1_2);

        ScsData<float, int> M_big_lp_scs_1_2;
        ScsData<double, int> M_big_hp_scs_1_2;

        convert_to_scs<float, int>(1.0, &lp_M_big, 1, 2, &M_big_lp_scs_1_2);
        convert_to_scs<double, int>(1.0, &hp_M_big, 1, 2, &M_big_hp_scs_1_2);

        M_big_lp_scs_1_2^exp_M_big_lp_scs_1_2;
        REQUIRE((M_big_lp_scs_1_2 == exp_M_big_lp_scs_1_2));

        M_big_hp_scs_1_2^exp_M_big_hp_scs_1_2;
        REQUIRE((M_big_hp_scs_1_2 == exp_M_big_hp_scs_1_2));

        ///// Second phase of test. Single proc, so many args don't matter
        exp_M_big_lp_scs_1_2_compressed.assign_explicit_test_data(&explicit_exp_M_big_lp_scs_1_2_compressed);
        exp_M_big_hp_scs_1_2_compressed.assign_explicit_test_data(&explicit_exp_M_big_hp_scs_1_2_compressed);

        ScsData<double, int> M_big_scs_1_2;
        convert_to_scs<double, int>(1.0, &M_big, 1, 2, &M_big_scs_1_2);
        int work_sharing_arr[2] = {0,10};
        std::string value_type = "mp";
        std::vector<std::vector<int>> communication_recv_idxs;
        communication_recv_idxs.push_back(std::vector<int>());
        std::vector<int> recv_counts_cumsum;

        // Compress columns
        // Actually shouldn't do anything in the single process case
        collect_local_needed_heri<double, int>(
            value_type,
            &communication_recv_idxs,
            &recv_counts_cumsum,
            &M_big_scs_1_2,
            &M_big_hp_scs_1_2,
            &M_big_lp_scs_1_2,
            work_sharing_arr,
            0,
            1
        );

        M_big_lp_scs_1_2^exp_M_big_lp_scs_1_2_compressed;
        REQUIRE((M_big_lp_scs_1_2 == exp_M_big_lp_scs_1_2_compressed));

        M_big_hp_scs_1_2^exp_M_big_hp_scs_1_2_compressed;
        REQUIRE((M_big_hp_scs_1_2 == exp_M_big_hp_scs_1_2_compressed));
    }
    SECTION("Test scs conversion with COO Matrix M_big, c=1, s=128"){
        std::cout << "Test " << ++test_cntr << ": M_big convert_to_scs_1_128 and compress..." << std::endl;

        MtxData<double, int> hp_M_big;
        MtxData<float, int> lp_M_big;

        // First, validate splitting routine
        seperate_lp_from_hp<double,int>(1.0, &M_big, &hp_M_big, &lp_M_big);

        lp_M_big^exp_M_big_lp;
        REQUIRE((lp_M_big == exp_M_big_lp));

        hp_M_big^exp_M_big_hp;
        REQUIRE((hp_M_big == exp_M_big_hp));

        // Then validate hp and lp COO structs convert sell_1_2 structs correctly
        exp_M_big_lp_scs_1_128.assign_explicit_test_data(&explicit_exp_M_big_lp_scs_1_128);
        exp_M_big_hp_scs_1_128.assign_explicit_test_data(&explicit_exp_M_big_hp_scs_1_128);

        ScsData<float, int> M_big_lp_scs_1_128;
        ScsData<double, int> M_big_hp_scs_1_128;

        convert_to_scs<float, int>(1.0, &lp_M_big, 1, 128, &M_big_lp_scs_1_128);
        convert_to_scs<double, int>(1.0, &hp_M_big, 1, 128, &M_big_hp_scs_1_128);

        M_big_lp_scs_1_128^exp_M_big_lp_scs_1_128;
        REQUIRE((M_big_lp_scs_1_128 == exp_M_big_lp_scs_1_128));

        M_big_hp_scs_1_128^exp_M_big_hp_scs_1_128;
        REQUIRE((M_big_hp_scs_1_128 == exp_M_big_hp_scs_1_128));

        ///// Second phase of test. Single proc, so many args don't matter
        exp_M_big_lp_scs_1_128_compressed.assign_explicit_test_data(&explicit_exp_M_big_lp_scs_1_128_compressed);
        exp_M_big_hp_scs_1_128_compressed.assign_explicit_test_data(&explicit_exp_M_big_hp_scs_1_128_compressed);

        ScsData<double, int> M_big_scs_1_128;
        convert_to_scs<double, int>(1.0, &M_big, 1, 128, &M_big_scs_1_128);
        int work_sharing_arr[2] = {0,10};
        std::string value_type = "mp";
        std::vector<std::vector<int>> communication_recv_idxs;
        communication_recv_idxs.push_back(std::vector<int>());
        std::vector<int> recv_counts_cumsum;

        // Test before too, because column compression shouldn't do anything here
        M_big_lp_scs_1_128^exp_M_big_lp_scs_1_128_compressed;
        REQUIRE((M_big_lp_scs_1_128 == exp_M_big_lp_scs_1_128_compressed));

        M_big_hp_scs_1_128^exp_M_big_hp_scs_1_128_compressed;
        REQUIRE((M_big_hp_scs_1_128 == exp_M_big_hp_scs_1_128_compressed));

        // Compress columns
        // Actually shouldn't do anything in the single process case
        collect_local_needed_heri<double, int>(
            value_type,
            &communication_recv_idxs,
            &recv_counts_cumsum,
            &M_big_scs_1_128,
            &M_big_hp_scs_1_128,
            &M_big_lp_scs_1_128,
            work_sharing_arr,
            0,
            1
        );

        M_big_lp_scs_1_128^exp_M_big_lp_scs_1_128_compressed;
        REQUIRE((M_big_lp_scs_1_128 == exp_M_big_lp_scs_1_128_compressed));

        M_big_hp_scs_1_128^exp_M_big_hp_scs_1_128_compressed;
        REQUIRE((M_big_hp_scs_1_128 == exp_M_big_hp_scs_1_128_compressed));
    }
}

// TEST_CASE("Validate function 'convert_to_scs' for sigma > 1 and c > 1 ", "[require]"){
//     SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){
        
//     }

TEST_CASE("Validate function 'collect_local_needed_heri' for c = 1 and sigma = 1 fixed", "[require]"){
    SECTION("Test column compression with SCS Matrices M1_scs, M1_hp_scs and M1_lp_scs, c=1, s=1 (i.e. crs), \
    assuming 2 MPI procs"){
        std::cout << "Test " << ++test_cntr << ": 2 proc M1 convert_to_scs..." << std::endl;
        const std::string value_type = "mp";

        // Artificially separate work over two processes
        int work_sharing_arr[3] = {0,1,3};

        // Proc 0 
        // Row compression of process local COO struct
        std::vector<int> p0_M1_local_row_coords(p0_M1.nnz, 0);

        for (int i = 0; i < p0_M1.nnz; ++i)
        {
            p0_M1_local_row_coords[i] = p0_M1.I[i] - p0_M1.I[0];
        }
        p0_M1.I = p0_M1_local_row_coords;

        // Separate lp and hp COO structs (assuming threshold = 1.0)
        MtxData<double, int> p0_hp_M1;
        MtxData<float, int> p0_lp_M1;
        seperate_lp_from_hp<double,int>(1.0, &p0_M1, &p0_hp_M1, &p0_lp_M1);

        // Convert local_mtx to SCS (CRS here) (TODO: probably not necessary, but convenient in current implementation)
        ScsData<double, int> p0_M1_scs_1_1;
        convert_to_scs<double, int>(1.0, &p0_M1, 1, 1, &p0_M1_scs_1_1);

        // Convert lp and hp to SCS (CRS here) structs
        ScsData<double, int> p0_hp_M1_scs_1_1;
        ScsData<float, int> p0_lp_M1_scs_1_1;
        convert_to_scs<double, int>(1.0, &p0_hp_M1, 1, 1, &p0_hp_M1_scs_1_1, work_sharing_arr, 0); 
        convert_to_scs<float, int>(1.0, &p0_lp_M1, 1, 1, &p0_lp_M1_scs_1_1, work_sharing_arr, 0);

        // Unused comm info, just needed for "collect_local_needed_heri"
        ///////////////////////////////////////////////////////////////////////////
        std::vector<std::vector<int>> p0_communication_recv_idxs;
        std::vector<std::vector<int>> p0_communication_send_idxs;

        // Fill vectors with empty vectors, representing places to store "to_send_idxs"
        for(int i = 0; i < 2; ++i){
            p0_communication_recv_idxs.push_back(std::vector<int>());
            p0_communication_send_idxs.push_back(std::vector<int>());
        }

        std::vector<int> p0_recv_counts_cumsum(2 + 1, 0);
        std::vector<int> p0_send_counts_cumsum(2 + 1, 0);
        ///////////////////////////////////////////////////////////////////////////

        // Column compression of all precision structs at once
        collect_local_needed_heri<double, int>(
            value_type, &p0_communication_recv_idxs, &p0_recv_counts_cumsum,
            &p0_M1_scs_1_1, &p0_hp_M1_scs_1_1, &p0_lp_M1_scs_1_1, work_sharing_arr, 0, 2
        );

        // Compare with expected structures
        // Compare column compression of entire local_scs struct
        exp_p0_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p0_M1_scs_1_1);
#ifdef DEBUG_MODE
        p0_M1_scs_1_1.print();
        exp_p0_M1_scs_1_1.print();
#endif
        p0_M1_scs_1_1^exp_p0_M1_scs_1_1;
        REQUIRE((p0_M1_scs_1_1 == exp_p0_M1_scs_1_1));
        // Compare column compression of entire hp local_scs struct
#ifdef DEBUG_MODE
        p0_hp_M1_scs_1_1.print();
        exp_p0_hp_M1_scs_1_1.print();
#endif
        exp_p0_hp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p0_hp_M1_scs_1_1);
        p0_hp_M1_scs_1_1^exp_p0_hp_M1_scs_1_1;
        REQUIRE((p0_hp_M1_scs_1_1 == exp_p0_hp_M1_scs_1_1));
        // // Compare column compression of entire lp local_scs struct
        exp_p0_lp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p0_lp_M1_scs_1_1);
#ifdef DEBUG_MODE
        p0_lp_M1_scs_1_1.print();
        exp_p0_lp_M1_scs_1_1.print();
#endif
        p0_lp_M1_scs_1_1^exp_p0_lp_M1_scs_1_1;
        REQUIRE((p0_lp_M1_scs_1_1 == exp_p0_lp_M1_scs_1_1));


        // Proc 1
        // Row compression of process local COO struct
        std::vector<int> p1_M1_local_row_coords(p1_M1.nnz, 0);

        for (int i = 0; i < p1_M1.nnz; ++i)
        {
            p1_M1_local_row_coords[i] = p1_M1.I[i] - p1_M1.I[0];
        }
        p1_M1.I = p1_M1_local_row_coords;

        // Separate lp and hp COO structs (assuming threshold = 1.0)
        MtxData<double, int> p1_hp_M1;
        MtxData<float, int> p1_lp_M1;
        seperate_lp_from_hp<double,int>(1.0, &p1_M1, &p1_hp_M1, &p1_lp_M1);

        // Convert local_mtx to SCS (CRS here) (TODO: probably not necessary, but convenient in current implementation)
        ScsData<double, int> p1_M1_scs_1_1;
        // std::cout << "p1_M1_scs_1_1:" << std::endl;
        convert_to_scs<double, int>(1.0, &p1_M1, 1, 1, &p1_M1_scs_1_1);

        // Convert lp and hp to SCS (CRS here) structs
        ScsData<double, int> p1_hp_M1_scs_1_1;
        ScsData<float, int> p1_lp_M1_scs_1_1;
        // std::cout << "p1_hp_M1_scs_1_1:" << std::endl;
        convert_to_scs<double, int>(1.0, &p1_hp_M1, 1, 1, &p1_hp_M1_scs_1_1, work_sharing_arr, 1); 
        // std::cout << "p1_lp_M1_scs_1_1:" << std::endl;
        convert_to_scs<float, int>(1.0, &p1_lp_M1, 1, 1, &p1_lp_M1_scs_1_1, work_sharing_arr, 1);

        // Unused comm info, just needed for "collect_local_needed_heri"
        ///////////////////////////////////////////////////////////////////////////
        std::vector<std::vector<int>> p1_communication_recv_idxs;
        std::vector<std::vector<int>> p1_communication_send_idxs;

        // Fill vectors with empty vectors, representing places to store "to_send_idxs"
        for(int i = 0; i < 2; ++i){
            p1_communication_recv_idxs.push_back(std::vector<int>());
            p1_communication_send_idxs.push_back(std::vector<int>());
        }

        std::vector<int> p1_recv_counts_cumsum(2 + 1, 0);
        std::vector<int> p1_send_counts_cumsum(2 + 1, 0);
        ///////////////////////////////////////////////////////////////////////////

        // Column compression of all precision structs at once
        collect_local_needed_heri<double, int>(
            value_type, &p1_communication_recv_idxs, &p1_recv_counts_cumsum,
            &p1_M1_scs_1_1, &p1_hp_M1_scs_1_1, &p1_lp_M1_scs_1_1, work_sharing_arr, 0, 2
        );

        // Compare with expected structures
        // Compare column compression of entire local_scs struct
        exp_p1_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p1_M1_scs_1_1);
#ifdef DEBUG_MODE
        p1_M1_scs_1_1.print();
        exp_p1_M1_scs_1_1.print();
#endif
        p1_M1_scs_1_1^exp_p1_M1_scs_1_1;
        REQUIRE((p1_M1_scs_1_1 == exp_p1_M1_scs_1_1));
        // Compare column compression of entire hp local_scs struct
        exp_p1_hp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p1_hp_M1_scs_1_1);
#ifdef DEBUG_MODE
        p1_hp_M1_scs_1_1.print();
        exp_p1_hp_M1_scs_1_1.print();
#endif
        p1_hp_M1_scs_1_1^exp_p1_hp_M1_scs_1_1;
        REQUIRE((p1_hp_M1_scs_1_1 == exp_p1_hp_M1_scs_1_1));
        // Compare column compression of entire lp local_scs struct
        exp_p1_lp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_p1_lp_M1_scs_1_1);
#ifdef DEBUG_MODE
        p1_lp_M1_scs_1_1.print();
        exp_p1_lp_M1_scs_1_1.print();
#endif
        p1_lp_M1_scs_1_1^exp_p1_lp_M1_scs_1_1;
        REQUIRE((p1_lp_M1_scs_1_1 == exp_p1_lp_M1_scs_1_1));
    }
}


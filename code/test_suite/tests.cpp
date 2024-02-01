#include "catch.hpp"
#include "classes_structs.hpp"
#include "mpi_funcs.hpp"
#include "test_data.hpp"

TEST_CASE("Validate COO matrix splitting function 'seperate_lp_from_hp' ", "[require]"){
    SECTION("Test splitting with COO Matrix M1"){
        std::cout << "Test 1..." << std::endl;

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
        std::cout << "Test 2..." << std::endl;

        exp_M1_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_scs_1_1);

        ScsData<double, int> M1_scs_1_1;

        convert_to_scs<double, int>(&M1, 1, 1, &M1_scs_1_1);

        M1_scs_1_1^exp_M1_scs_1_1;
        REQUIRE((M1_scs_1_1 == exp_M1_scs_1_1));
    }
    SECTION("Test scs conversion with split COO Matrix exp_M1_hp, c=1, s=1"){
        std::cout << "Test 3..." << std::endl;

        exp_M1_hp_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_hp_scs_1_1);

        ScsData<double, int> M1_hp_scs_1_1;

        convert_to_scs<double, int>(&exp_M1_hp, 1, 1, &M1_hp_scs_1_1);

        M1_hp_scs_1_1^exp_M1_hp_scs_1_1;
        REQUIRE((M1_hp_scs_1_1 == exp_M1_hp_scs_1_1));
    }
    SECTION("Test scs conversion with split COO Matrix exp_M1_lp, c=1, s=1"){
        std::cout << "Test 4..." << std::endl;

        exp_M1_lp_scs_1_1.assign_explicit_test_data(&explicit_exp_M1_lp_scs_1_1);

        ScsData<float, int> M1_lp_scs_1_1;

        convert_to_scs<float, int>(&exp_M1_lp, 1, 1, &M1_lp_scs_1_1);

        M1_lp_scs_1_1^exp_M1_lp_scs_1_1;
        REQUIRE((M1_lp_scs_1_1 == exp_M1_lp_scs_1_1));
    }

// TEST_CASE("Validate function 'convert_to_scs' for c > 1 and sigma = 1 fixed", "[require]"){
//     SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){

//     }

// TEST_CASE("Validate function 'convert_to_scs' for sigma > 1 and c = 1 fixed", "[require]"){
//     SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){
        
//     }

// TEST_CASE("Validate function 'convert_to_scs' for sigma > 1 and c > 1 ", "[require]"){
//     SECTION("Test scs conversion with COO Matrix M1, c=1, s=1 (i.e. crs)"){
        
//     }
}
#include "catch.hpp"
#include "classes_structs.hpp"
#include "mpi_funcs.hpp"
#include "test_data.hpp"

TEST_CASE("Validate COO matrix splitting function 'seperate_lp_from_hp' ", "[require]"){
    SECTION("Test splitting with COO Matrix M1"){
        REQUIRE(test());
    }
}


// TEST_CASE("Validate COO matrix splitting function 'seperate_lp_from_hp' ", "[require]"){
//     SECTION("Test splitting with COO Matrix M1"){

        // // Declare structs to be tested
        // MtxData<double, int> M1_hp;
        // MtxData<float, int> M1_lp;

        // seperate_lp_from_hp<double, int>(&M1_mp, &M1_hp, &M1_lp);
        // M1_hp^exp_M1_hp;
        // REQUIRE((M1_hp == exp_M1_hp));
        // M1_lp^exp_M1_lp;
        // REQUIRE((M1_lp == exp_M1_lp));
//     }
// }
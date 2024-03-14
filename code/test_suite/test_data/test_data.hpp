#ifndef TESTDATA_H
#define TESTDATA_H

#include "classes_structs.hpp"

//////////////////////////// M0 test data ////////////////////////////
extern MtxData<double, int> M0;

//////////////////////////// M1 test data ////////////////////////////
extern MtxData<double, int> M1;

///////////////////// M1 with p0, and p1 test data /////////////////////
extern MtxData<double, int> p0_M1;
extern MtxData<double, int> p1_M1;

extern ScsData<double, int> exp_p0_M1_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_p0_M1_scs_1_1;
extern ScsData<double, int> exp_p0_hp_M1_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_p0_hp_M1_scs_1_1;
extern ScsData<float, int> exp_p0_lp_M1_scs_1_1;
extern ScsExplicitData<float, int> explicit_exp_p0_lp_M1_scs_1_1;
extern ScsData<double, int> exp_p1_M1_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_p1_M1_scs_1_1;
extern ScsData<double, int> exp_p1_hp_M1_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_p1_hp_M1_scs_1_1;
extern ScsData<float, int> exp_p1_lp_M1_scs_1_1;
extern ScsExplicitData<float, int> explicit_exp_p1_lp_M1_scs_1_1;

//////////////////////////// M1_scs test data ////////////////////////////
extern ScsData<double, int> exp_M1_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_scs_1_1;

//////////////////////////// M1_scs_te_row test data ////////////////////////////
extern MtxData<double, int> M1_te_row;
extern ScsData<double, int> exp_M1_te_row_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_te_row_scs_1_1;

//////////////////////////// M1_scs_me_row test data ////////////////////////////
extern MtxData<double, int> M1_me_row;
extern ScsData<double, int> exp_M1_me_row_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_me_row_scs_1_1;

//////////////////////////// M1_scs_be_row test data ////////////////////////////
extern MtxData<double, int> M1_be_row;
extern ScsData<double, int> exp_M1_be_row_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_be_row_scs_1_1;

//////////////////////////// M1_scs_le_col test data ////////////////////////////
extern MtxData<double, int> M1_le_col;
extern ScsData<double, int> exp_M1_le_col_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_le_col_scs_1_1;

//////////////////////////// M1_scs_me_col test data ////////////////////////////
extern MtxData<double, int> M1_me_col;
extern ScsData<double, int> exp_M1_me_col_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_me_col_scs_1_1;

//////////////////////////// M1_scs_be_row_re_col test data ////////////////////////////
extern MtxData<double, int> M1_be_row_re_col;
extern ScsData<double, int> exp_M1_be_row_re_col_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_be_row_re_col_scs_1_1;

//////////////////////////// M1-hp test data ////////////////////////////
extern MtxData<double, int> exp_M1_hp;
extern ScsData<double, int> exp_M1_hp_scs_1_1;
extern ScsExplicitData<double, int> explicit_exp_M1_hp_scs_1_1;

//////////////////////////// M1-lp test data ////////////////////////////
extern MtxData<float, int> exp_M1_lp;
extern ScsData<float, int> exp_M1_lp_scs_1_1;
extern ScsExplicitData<float, int> explicit_exp_M1_lp_scs_1_1;

//////////////////////////// M-big test data ////////////////////////////
extern MtxData<double, int> M_big;

//////////////////////////// M-big-hp test data ////////////////////////////
extern MtxData<float, int> exp_M_big_lp;
extern ScsData<float, int> exp_M_big_lp_scs_1_2;
extern ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_2;
extern ScsData<float, int> exp_M_big_lp_scs_1_2_compressed;
extern ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_2_compressed;
extern ScsData<float, int> exp_M_big_lp_scs_1_128;
extern ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_128;
extern ScsData<float, int> exp_M_big_lp_scs_1_128_compressed;
extern ScsExplicitData<float, int> explicit_exp_M_big_lp_scs_1_128_compressed;

//////////////////////////// M-big-lp test data ////////////////////////////
extern MtxData<double, int> exp_M_big_hp;
extern ScsData<double, int> exp_M_big_hp_scs_1_2;
extern ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_2;
extern ScsData<double, int> exp_M_big_hp_scs_1_2_compressed;
extern ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_2_compressed;
extern ScsData<double, int> exp_M_big_hp_scs_1_128;
extern ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_128;
extern ScsData<double, int> exp_M_big_hp_scs_1_128_compressed;
extern ScsExplicitData<double, int> explicit_exp_M_big_hp_scs_1_128_compressed;

#endif
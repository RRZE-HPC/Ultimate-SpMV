declare -a matrices=(
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_5.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_6.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_7.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_8.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_95.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_98.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_l0_0_9.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_5.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_6.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_7.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_8.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_95.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_98.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_mag_0_9.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_5.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_6.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_7.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_8.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_95.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_98.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_rand_0_9.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_5.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_6.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_7.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_8.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_95.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_98.mtx"
   "/home/vault/k107ce/k107ce17/bench_matrices/transformer/mhead_attention_var_0_9.mtx"
)

for matrix in "${matrices[@]}";
do
    python mm2sparsityPattern.py $matrix 0
done
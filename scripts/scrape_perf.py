
with open('spmv_bench_AP_32_1024.txt') as file:
    Lines=file.readlines()
    line_count = 0
    for line in Lines:
        if "Total Gflops:" in line:
            print(Lines[line_count + 2].split()[0], end=",\n")
        line_count += 1
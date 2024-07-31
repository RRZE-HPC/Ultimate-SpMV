import pprint

bandwidths = {}
data_vols = {}

with open('/home/hpc/k107ce/k107ce17/linking_it_solve/iterative_solvers/ML_Geer_data_vol_tests_dffix/AP_ML_Geer_epsilon_53.txt') as file:
    Lines=file.readlines()
    line_count = 0
    for line in Lines:
        if "Region Info" in line:
            thread_count = line.split()[-2]
            if thread_count not in bandwidths.keys():
                bandwidths[thread_count] = []
            if thread_count not in data_vols.keys():
                data_vols[thread_count] = []
            try:
                bandwidths[thread_count].append(float(Lines[line_count + 82].split()[6]))
                data_vols[thread_count].append(float(Lines[line_count + 83].split()[7]))
            except:
                bandwidths[thread_count].append("Fill manually")
                data_vols[thread_count].append("Fill manually") 
        line_count += 1

print("bandwidths")
pprint.pprint(bandwidths)
print("Data volume")
pprint.pprint(data_vols)
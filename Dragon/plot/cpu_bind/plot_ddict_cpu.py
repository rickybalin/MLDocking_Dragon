import numpy as np
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

case_files = [
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/no_bind/mldocking_seq_ddict.o5514950', #mldocking_seq_ddict.o5509642',
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/1c_1s/mldocking_seq_ddict.o5504843',
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/2c_1s/mldocking_seq_ddict.o5504912',
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/2c_2s/mldocking_seq_ddict.o5509551',
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/4c_1s/mldocking_seq_ddict.o5509066',
    #'/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/4c_1s_2man/mldocking_seq_ddict.o5509066',
    '/flare/hpe_dragon_collab/balin/PASC25/runs/ddict_cpu_bind/tiny_8192/4c_2s/mldocking_seq_ddict.o5509012'
]
legend = [
    'no binding',
    '1 core on 1 socket',
    '2 cores on 1 socket',
    '4 cores on 2 socket',
    '4 cores on 1 socket',
    #'4 cores on 1 socket 2 managers',
    '8 cores on 2 socket',
]
keys = ['nodes',
        'procs',
        'load_IO_avg',
        'load_IO_max',
        'load_ddict_avg',
        'load_ddict_max',
        'inf_ddict_avg',
        'inf_ddict_max',
        'sort_IO',
        'load',
        'inference',
        'sort']


def parse_history_line(l):
    vals = []
    parsed_l = l.split(' ')
    for i in range(len(parsed_l)):
        if parsed_l[i] == 'mse:':
            vals.append(float(parsed_l[i+1]))
        if parsed_l[i] == 'r2:':
            vals.append(float(parsed_l[i+1]))
        if parsed_l[i] == 'val_mse:':
            vals.append(float(parsed_l[i+1]))
        if parsed_l[i] == 'val_r2:':
            vals.append(float(parsed_l[i+1]))
    return vals

cases = {}
for case in case_files:
    cases[case] = {}
    for label in keys:
        cases[case][label] = []
    with open(case,'r') as fh:
        inf_ddict_time = []
        for l in fh:
            if "Launched Dragon Dictionary for inference" in l:
                cases[case]['nodes'] = int(l.split(' ')[-2])
            if "Number of Pool processes" in l:
                cases[case]['procs'] = int(l.split(' ')[-1])
            if "Loaded inference data in" in l:
                cases[case]['load'] = float(l.split(' ')[-2])
            if "IO times:" in l:
                parsed_l=l.split(":")[-1].split(",")
                cases[case]['load_IO_avg'] = float(parsed_l[0].split("=")[-1].split(" ")[0])
                cases[case]['load_IO_max'] = float(parsed_l[1].split("=")[-1].split(" ")[0])
            if "DDict times:" in l:
                parsed_l=l.split(":")[-1].split(",")
                cases[case]['load_ddict_avg'] = float(parsed_l[0].split("=")[-1].split(" ")[0])
                cases[case]['load_ddict_max'] = float(parsed_l[1].split("=")[-1].split(" ")[0])
            if "Performed inference in" in l:
                cases[case]['inference'] = float(l.split(' ')[-3])
            if "Performed inference on" in l:
                parsed_l = l.split(":")[-1].split(",")[1]
                inf_ddict_time.append(float(parsed_l.split("=")[-1]))
            if "Performed sorting of 10000 compounds" in l:
                cases[case]['sort_IO'] = float(l.split(":")[-1].split(",")[-1].split("=")[-1])
            if "Performed sorting of 8192" in l:
                cases[case]['sort'] = float(l.split(" ")[-3])


        cases[case]['inf_ddict_avg'] = sum(inf_ddict_time)/len(inf_ddict_time)
        cases[case]['inf_ddict_max'] = max(inf_ddict_time)


# Plot load, inf, sort times
labels = ['Data Loader', 'Inference', 'Sorting']
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars
factors = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
for i, case in enumerate(cases):
    axs.bar(x+factors[i]*width, [cases[case]['load'], cases[case]['inference'], cases[case]['sort']], width,label=legend[i])
axs.set_yscale('log')
axs.set_ylabel('Time [sec]')
axs.set_title('Component Time')
axs.set_xticks(x);axs.set_xticklabels(labels)
#axs.set_xticks(x, labels)
axs.legend()
axs.grid(axis='y')
fig.savefig('plt_comp_time.png')

# Plot DDIct times
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
for i, case in enumerate(cases):
    axs.bar(x+factors[i]*width, [cases[case]['load_IO_avg'], cases[case]['inf_ddict_avg'], cases[case]['sort_IO']], width,label=legend[i])
axs.set_ylabel('Time [sec]')
axs.set_title('Component DDict Time')
axs.set_xticks(x);axs.set_xticklabels(labels)
#axs.set_xticks(x, labels)
axs.legend()
axs.grid(axis='y')
fig.savefig('plt_ddict_time.png')

"""
# Plot processes
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
for i, case in enumerate(cases.keys()):
    axs.plot(cases[case]['nodes'],cases[case]['load_IO_avg'],label=labels[i],ls="-",linewidth=2)
#axs.set_yscale("log")
axs.grid()
axs.set_xlabel('Nodes')
axs.set_ylabel('IO')
axs.legend(loc='lower right')
fig.savefig('plt_pool_proc.png')
"""

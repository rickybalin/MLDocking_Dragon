import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)


class DDictScaling:
    def __init__(self,path,node_list):
        self.base_path = path
        self.node_list = node_list
        self.n_nodes = len(node_list)
        self.stats = {
            'inf_ddict_init': np.zeros((self.n_nodes,)),
            'model_ddict_init': np.zeros((self.n_nodes,)),
            'pool_procs': np.zeros((self.n_nodes,)),
            'num_files': np.zeros((self.n_nodes,)),
            'load_time': np.zeros((self.n_nodes,)),
            'load_IO_avg_time': np.zeros((self.n_nodes,)),
            'load_IO_max_time': np.zeros((self.n_nodes,)),
            'load_ddict_avg_time': np.zeros((self.n_nodes,)),
            'load_ddict_max_time': np.zeros((self.n_nodes,)),
            'load_mem': np.zeros((self.n_nodes,)),
            'load_imbalance_keys': np.zeros((self.n_nodes,)),
            'load_imbalance_mem': np.zeros((self.n_nodes,)),
        }
        self.counts = {
            'inf_ddict_init': np.zeros((self.n_nodes,)),
            'model_ddict_init': np.zeros((self.n_nodes,)),
            'pool_procs': np.zeros((self.n_nodes,)),
            'num_files': np.zeros((self.n_nodes,)),
            'load_time': np.zeros((self.n_nodes,)),
            'load_IO_avg_time': np.zeros((self.n_nodes,)),
            'load_IO_max_time': np.zeros((self.n_nodes,)),
            'load_ddict_avg_time': np.zeros((self.n_nodes,)),
            'load_ddict_max_time': np.zeros((self.n_nodes,)),
            'load_mem': np.zeros((self.n_nodes,)),
            'load_imbalance_keys': np.zeros((self.n_nodes,)),
            'load_imbalance_mem': np.zeros((self.n_nodes,)),
        }

    def avg(self, my_list):
        return sum(my_list)/len(my_list)

    def parse_files(self):
        for i in range(self.n_nodes):
            path = self.base_path+f"/{self.node_list[i]}/mldocking*"
            # Loop over runs found
            run_files = glob.glob(path)
            for run_file in run_files:
                print('Reading file: ', run_file)
                with open(run_file,'r') as fh:
                    ddict_keys = []
                    ddict_mem = []
                    for l in fh:
                        if "Launched Dragon Dictionary for inference" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['inf_ddict_init'][i] += float(l.split(' ')[-2])
                            self.counts['inf_ddict_init'][i] += 1
                        if "Launched Dragon Dictionary for model" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['model_ddict_init'][i] += float(l.split(' ')[-2])
                            self.counts['model_ddict_init'][i] += 1
                        if "Number of files to read is" in l:
                            self.stats['num_files'][i] += int(l.split(' ')[-1])
                            self.counts['num_files'][i] += 1
                        if "Number of Pool processes" in l:
                            self.stats['pool_procs'][i] += int(l.split(' ')[5])
                            self.counts['pool_procs'][i] += 1
                        if "Loaded inference data in" in l:
                            if "\n" in l:
                                l = l.replace("\n","")
                            self.stats['load_time'][i] += float(l.split(' ')[4])
                            self.counts['load_time'][i] += 1
                        if "IO times:" in l:
                            parsed_l=l.split(":")[-1].split(",")
                            self.stats['load_IO_avg_time'][i] += float(parsed_l[0].split("=")[-1].split(" ")[0])
                            self.counts['load_IO_avg_time'][i] += 1
                            self.stats['load_IO_max_time'][i] += float(parsed_l[1].split("=")[-1].split(" ")[0])
                            self.counts['load_IO_max_time'][i] += 1
                        if "DDict times:" in l:
                            parsed_l=l.split(":")[-1].split(",")
                            self.stats['load_ddict_avg_time'][i] += float(parsed_l[0].split("=")[-1].split(" ")[0])
                            self.counts['load_ddict_avg_time'][i] += 1
                            self.stats['load_ddict_max_time'][i] += float(parsed_l[1].split("=")[-1].split(" ")[0])
                            self.counts['load_ddict_max_time'][i] += 1
                        if "manager_ID" in l:
                            ddict_keys.append(int(l.split(',')[2].split('=')[-1]))
                            mem_frac = float(l.split(',')[3].split('=')[-1])/100
                            mem_tot = float(l.split(',')[4].split('=')[-1])
                            ddict_mem.append(mem_frac*mem_tot)

                self.stats['load_mem'][i] = self.avg(ddict_mem)
                self.counts['load_mem'][i] += 1
                self.stats['load_imbalance_keys'][i] = (max(ddict_keys) - self.avg(ddict_keys)) / (self.avg(ddict_keys))
                self.counts['load_imbalance_keys'][i] += 1
                self.stats['load_imbalance_mem'][i] = (max(ddict_mem) - self.avg(ddict_mem)) / (self.avg(ddict_mem))
                self.counts['load_imbalance_mem'][i] += 1


        # Divide by the counts for each gpu number to get the average over runs
        for key in self.stats.keys():
            val = self.stats[key]
            if len(val.shape)==1:
                #self.train_fom[key] = np.divide(val, self.counts[key], where=self.counts[key]>0)
                np.divide(val, self.counts[key], out=self.stats[key], where=self.counts[key]>0)
            else:
                for j in range(val.shape[1]):
                    #self.train_fom[key][:,j] = np.divide(val[:,j], self.counts[key], where=self.counts[key]>0)
                    np.divide(val[:,j], self.counts[key], out=self.stats[key][:,j], where=self.counts[key]>0)


base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/loading/strong_scale_ck64'
node_list = [4,8,16,32,64,128]
chk64 = DDictScaling(base_path,node_list)
chk64.parse_files()

base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/loading/strong_scale_ck1'
node_list = [8,16,32]
chk1 = DDictScaling(base_path,node_list)
chk1.parse_files()

base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/loading/strong_scale_ch64_full'
node_list = [16,32,64,128,256]
chk64_full = DDictScaling(base_path,node_list)
chk64_full.parse_files()

base_path = '/lus/flare/projects/hpe_dragon_collab/balin/PASC25/runs/loading/weak_scale_ck64'
node_list = [8,32,128]
chk64_w = DDictScaling(base_path,node_list)
chk64_w.parse_files()

# Plot DDict init times
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(chk64.node_list, chk64.stats['inf_ddict_init'],label = "Inference DDict",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(chk64.node_list, chk64.stats['model_ddict_init'],label = "Model DDict",marker="o",ls="-",markersize=10, linewidth=2)
#axs.set_xscale("log")
axs.grid()
axs.set_xlabel('Number of Nodes')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='lower left')
axs.set_ylabel('Time [sec]')
axs.set_title('DDict Init Time')
fig.tight_layout(pad=2.0)
fig.savefig("plt_ddict_strong_scale.png")

# Plot DDict memory usage and load imbalance
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(chk64.node_list, chk64.stats['load_mem']/1000/1000/1000,label = "Average Data Utilization Per Node",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [chk64.stats['load_mem'][0]/(1000*1000*1000)/(nodes/chk64.node_list[0]) for nodes in chk64.node_list]
axs.plot(chk64.node_list, ideal,ls="--", linewidth=1, color="k")

ax2 = axs.twinx()
ax2.plot(chk64.node_list, chk64.stats['load_imbalance_mem']*100,label = "Memory Imbalance",marker="o",ls="-",markersize=10, linewidth=2, color='r')
ax2.plot(chk64.node_list, chk64.stats['load_imbalance_keys']*100,label = "Key Imbalance",marker="o",ls="--",markersize=10, linewidth=2, color='r')
ax2.set_ylabel('Load Imbalance [%]', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0,10)
ax2.legend(loc='center right')

#axs.set_xscale("log")
axs.grid()
axs.set_xlabel('Number of Nodes')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='upper right')
axs.set_ylabel('Size [GB]')
axs.set_title('DDict Statistics')
fig.tight_layout(pad=2.0)
fig.savefig("plt_data_strong_scale.png")

# Plot data loader time
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
#axs.plot(chk1.node_list, chk1.stats['load_time'],label = "Chunksize 1",marker="o",ls="-",markersize=10, linewidth=2)
axs.plot(chk64.node_list, chk64.stats['load_time'],label = "131,072 files",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [chk64.stats['load_time'][0]/(nodes/chk64.node_list[0]) for nodes in chk64.node_list]
axs.plot(chk64.node_list, ideal,ls="--", linewidth=1, color="k")
chk64_efficiency = [id/real*100 for id,real in zip(ideal,chk64.stats['load_time'].tolist())]
for i in range(len(chk64_efficiency)):
    axs.text(chk64.node_list[i] + 0.1, chk64.stats['load_time'][i]+ 0.1, str(round(chk64_efficiency[i],1))+" %", fontsize=9, color='black')

axs.plot(chk64_full.node_list, chk64_full.stats['load_time'],label = "500,354 files",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [chk64_full.stats['load_time'][0]/(nodes/chk64_full.node_list[0]) for nodes in chk64_full.node_list]
axs.plot(chk64_full.node_list, ideal,ls="--", linewidth=1, color="k")
chk64_full_efficiency = [id/real*100 for id,real in zip(ideal,chk64_full.stats['load_time'].tolist())]
for i in range(len(chk64_full_efficiency)):
    axs.text(chk64_full.node_list[i] + 0.1, chk64_full.stats['load_time'][i]+ 0.1, str(round(chk64_full_efficiency[i],1))+" %", fontsize=9, color='black')

#axs.set_xscale("log")
axs.grid()
axs.set_xlabel('Number of Nodes')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='upper right')
axs.set_ylabel('Time [sec]')
axs.set_title('Data Loader Execution Time')
fig.tight_layout(pad=2.0)
fig.savefig("plt_load_strong_scale.png")


# Plot data loader efficiency 
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
ideal = [100 for _ in range(5000)]
axs.plot(list(range(5000)), ideal,ls="--", linewidth=1, color="k")
ch64_files_per_proc = chk64.stats["num_files"]/chk64.stats["pool_procs"]
axs.plot(ch64_files_per_proc, chk64_efficiency,label = "131,072 files",marker="o",ls="-",markersize=10, linewidth=2)
ch64_full_files_per_proc = chk64_full.stats["num_files"]/chk64_full.stats["pool_procs"]
axs.plot(ch64_full_files_per_proc, chk64_full_efficiency,label = "500,354 files",marker="o",ls="-",markersize=10, linewidth=2)
#axs.set_xscale("log")
axs.grid()
axs.set_xlabel('Number of Files per Process')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='lower right')
axs.set_ylabel('Scaling Efficiency [%]')
axs.set_ylim(0,110)
axs.set_title('Data Loader Efficiency')
fig.tight_layout(pad=2.0)
fig.savefig("plt_load_eff_strong_scale.png")

# Plot weak scaling
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
axs.plot(chk64_w.node_list, chk64_w.stats['load_time'],label = "weak scaling",marker="o",ls="-",markersize=10, linewidth=2)
ideal = [chk64_w.stats['load_time'][0] for _ in chk64_w.node_list]
axs.plot(chk64_w.node_list, ideal,ls="--", linewidth=1, color="k")
chk64_w_efficiency = [id/real*100 for id,real in zip(ideal,chk64_w.stats['load_time'].tolist())]
for i in range(len(chk64_w_efficiency)):
    axs.text(chk64_w.node_list[i] + 0.1, chk64_w.stats['load_time'][i]+ 0.1, str(round(chk64_w_efficiency[i],1))+" %", fontsize=9, color='black')
#axs.set_xscale("log")
axs.grid()
axs.set_xlabel('Number of Nodes')
#fig.legend(bbox_to_anchor=(1.25,0.7))
axs.legend(loc='lower right')
axs.set_ylabel('Time [sec]')
axs.set_ylim(0,110)
axs.set_title('Data Loader Execution Time')
fig.tight_layout(pad=2.0)
fig.savefig("plt_load_weak_scale.png")



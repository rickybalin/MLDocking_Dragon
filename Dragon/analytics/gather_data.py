from matplotlib import pyplot as plt
import glob
import numpy as np
import os

supplementry_tests = {"/grand/hpe_dragon_collab/csimpson/MLDocking_Dragon/Dragon/dragon0.91v2_data_loading/ml_docking_dragon_128.o1965282":
                        {"tiny": {"nodes": 128,
                                "size": sum([24954481223,26452474238,26777462247,26887719330])/(1024*1024*1024),
                                "Data Load Time (s)": 323.082,
                                },
                        "small": {"nodes": 128,
                                  "size": sum([100054476363,105821579496,107088144060,107663101019])/(1024*1024*1024),
                                  "Data Load Time (s)": 331.181,
                                  },
                        "med": {"nodes": 128,
                                "size": sum([400516724316,423145539453,428205020419,430748592352])/(1024*1024*1024),
                                "Data Load Time (s)": 391.671,
                                },
                        },
                    }


def gather_data(paths):
    datasets = paths.keys()
    data = {}
    inf_data = {}
    for d in datasets:
        data[d] = parse_stdout(paths[d])
        inf_data[d] = parse_inference_log(paths[d])
    return data, inf_data

def parse_inference_log(path_list):
    ret = {}    

    for p in path_list:

        tot_smiles = 0
        tot_data_moved = 0
        tot_data_move_time = 0
        tot_time = 0
        num_active_procs = 0
        

        nodes = p.split("/")[0]
        key = p.split("/")[-1]
        ret[key] = {}
        
        ret[key]["nodes"] = int(nodes[:-5])
        nodes = int(nodes[:-5])
        base_path = "/".join(p.split("/")[:-1])
        log_file = os.path.join(base_path,"infer_switch.log")
        if os.path.exists(log_file):
            with open(log_file,"r") as f:
                lines = f.readlines()
                for line in lines:
                    line_split = line.split()
                    if "num_smiles=" in line:
                        num_smiles = int(line_split[6].split("=")[1])
                        tot_smiles+= num_smiles
                        if num_smiles > 0:
                            num_active_procs += 1
                    if "total_time=" in line:
                        tot_time += float(line_split[6].split("=")[1])
                    if "data_move_time=" in line:
                        tot_data_move_time += float(line_split[6].split("=")[1])
                    if "data_move_size=" in line:
                        tot_data_moved += int(line_split[6].split("=")[1])/(1024*1024*1024)
        if tot_time > 0:
            ret[key]["data_move_time_frac"] = tot_data_move_time/tot_time
            ret[key]["smiles_per_s"] = tot_smiles/(tot_time/(4*nodes))
        if tot_smiles > 0:
            ret[key]["time_per_smiles"] = (tot_time/tot_smiles)/(4*nodes)
        if tot_data_move_time > 0:
            ret[key]["data_per_s"] = tot_data_moved/(tot_data_move_time/(4*nodes))
        ret[key]["tot_smiles"] = tot_smiles/(4*nodes)
        ret[key]["tot_time"] = tot_time/(nodes*4)
        ret[key]["num_active_procs"] = num_active_procs
        ret[key]["tot_data_moved"] = tot_data_moved/(4*nodes)
        ret[key]["tot_data_move_time"] = tot_data_move_time/(4*nodes)
    return ret

        
        
                




# def parse_inference_worker(lines):
   
#     tot_smiles = []
#     tot_data_moved = []
#     tot_data_move_time = []
#     ave_data_move_time_frac = []
#     for line in lines:
#         if "Performed inference on key" in line:
#             split_line = line.split()
#             key_time = float(split_line[5].split("=")[1])
#             len_smiles_sorted = int(split_line[6].split("=")[1])
#             key_data_moved_size = int(split_line[7].split("=")[1])/(1024*1024)
#             key_data_moved_size = float(split_line[8].split("=")[1])



# def parse_inference(path_list):
#     ret = {}

#     for p in path_list:
        

#         base_path = "/".join(p.split("/")[:-1])
#         worker_files = glob.glob(f"{base_path}/ws_worker_*.log")
#         for wf in worker_files:
#             with open(wf,"r") as f:
#                 lines = f.readlines()





def parse_stdout(path_list):
 
    ret = {}
    for p in path_list:
        nodes = p.split("/")[0]
        key = p.split("/")[-1]
        ret[key] = {}
        ret[key]["nodes"] = int(nodes[:-5])
        ret[key]["size"] = 0
        with open(p,"r") as f:
            lines = f.readlines()
            for line in lines:
                if "Total time " in line:
                    split_line = line.split()
                    total_time = split_line[2]
                    ret[key]["Total Time (s)"] = float(total_time)
                if "Size of dataset is " in line:
                    split_line = line.split()
                    line_size = int(split_line[4])/(1024*1024*1024)
                    ret[key]["size"] += line_size
                if "Loaded inference data in " in line:
                    split_line = line.split()
                    load_time = float(split_line[4])
                    ret[key]["Data Load Time (s)"] = float(load_time)
                if "Performed inference in " in line:
                    split_line = line.split()
                    inf_time = float(split_line[3])
                    ret[key]["Inference Time (s)"] = float(inf_time)
                if "num_files=" in line:
                    split_line = line.split("=")
                    num_files = int(split_line[1])
                    ret[key]["Number of Files"] = float(num_files)
            if "Number of Files" in ret[key].keys() and "Inference Time (s)" in ret[key].keys():
                ret[key]["Molecules per Second per node"] = ret[key]["Number of Files"]*1e5/ret[key]["Inference Time (s)"]/ret[key]["nodes"]
                ret[key]["Molecules per Second"] = ret[key]["Number of Files"]*1e5/ret[key]["Inference Time (s)"]
            if "Inference Time (s)" in ret[key].keys():
                ret[key]["Inference Time x Nodes"] = ret[key]["nodes"]*ret[key]["Inference Time (s)"]
    node_result_16 = [ret[k]["Inference Time x Nodes"] for k in ret.keys() if ret[k]['nodes'] == 16][0]
    for k in ret.keys():
       if "Inference Time x Nodes" in ret[k].keys():
           ret[k]["Inference Time x Nodes"] = ret[k]["Inference Time x Nodes"]/node_result_16
    return ret




def get_dataset_path(datasets, experiment="data_loading_inference"):

    ret_paths = {}
    for dataset in datasets:
        stdout_paths = glob.glob(f"*nodes/{dataset}*/ml_docking_dragon*.o*")

        node_dirs = []
        
        for p in stdout_paths:
            print(p)
            split_p = p.split("/")
            node_dirs.append(split_p[0])
        node_dirs = list(set(node_dirs))
        last_stdout = []
        for nd in node_dirs:
            match_paths = [p for p in stdout_paths if nd in p]
            
            match_paths.sort()
            last_stdout.append(match_paths[-1])
        ret_paths[dataset] = last_stdout
    return ret_paths

def plot_quantity_v_nodes(data, quantity, label = None, ax = None, xlog=False,ylog=False):
    keys = data.keys()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    #font = {'fontname': 'Times New Roman'}
    fig = plt.figure()
    datasets = data.keys()
    for d in datasets:
        ret = data[d]
        rkeys = ret.keys()
        x = []
        y = []  
        for k in rkeys:
            metrics = ret[k]
            mkeys = metrics.keys()
            print(f"{mkeys=}")
            test_data_presence = 0
            if "size" in mkeys:
                test_data_presence = metrics["size"]
            elif "tot_smiles" in mkeys:
                print("plotting inference")
                test_data_presence = metrics["tot_smiles"]
            if (test_data_presence > 0) and quantity in mkeys:
                x = np.append(x,metrics["nodes"])
                y = np.append(y,metrics[quantity])
        
        x = np.array(x)
        y = np.array(y)
        isort = np.argsort(x)
        x = x[isort]
        y = y[isort]

        if ax is None:
            plt.plot(x,y,'s--',label=d)
            if xlog:
                plt.xscale("log",base=2)
            if ylog:
                plt.yscale("log",base=2)
    plt.legend(loc=0)
    plt.xlabel("Node Count")
    plt.ylabel(quantity)
    quantity_str = "_".join(quantity.split())
    fig.tight_layout()
    fig.savefig(f"{quantity_str}_v_nodes.pdf")




paths = get_dataset_path(["test-tiny","test-med","test-large"])
data,inf_data = gather_data(paths)
plot_quantity_v_nodes(data,"Total Time (s)")
plot_quantity_v_nodes(data,"Data Load Time (s)")
plot_quantity_v_nodes(data,"Inference Time (s)")

plot_quantity_v_nodes(data,"Molecules per Second",ylog=True,xlog=True)
plot_quantity_v_nodes(data,"Molecules per Second per node",ylog=True,xlog=True)
plot_quantity_v_nodes(data,"Inference Time x Nodes",ylog=True,xlog=True)
#plot_quantity_v_nodes(inf_data,"smiles_per_s")
quantities = ["data_move_time_frac",
              "time_per_smiles",
              "data_per_s","tot_smiles","tot_time","num_active_procs",
              "tot_data_moved","tot_data_move_time","smiles_per_s"]
for k in quantities:
    plot_quantity_v_nodes(inf_data,k)
#plot_quantity_v_nodes(inf_data,"data_per_s")
#plot_quantity_v_nodes(inf_data,"tot_time")
#plot_quantity_v_nodes(inf_data,"time_per_smiles")

print(inf_data)
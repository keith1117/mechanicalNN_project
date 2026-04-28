from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from MulaTOVA_MNN import TopologyOptimizer
import matplotlib.pyplot as plt
from utils import PytorchMinMaxScaler, plot_latent,setDevice,set_seed
import torch
import time

## Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./struct.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name
print(config.vae_file_path)
overrideGPU = False
device = setDevice(overrideGPU) 
torch.autograd.set_detect_anomaly(True)

for config.example in [7]:
    plt.close('all') 
    start = time.perf_counter()
    topOpt = TopologyOptimizer(config)
    topOpt.optimizeDesign(config) 
    print("Time taken (secs): {:.2F}".format( time.perf_counter() - start))
    print(topOpt.exper_name)
    topOpt.plotConvergence() 

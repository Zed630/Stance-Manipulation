from ml_collections import config_dict
import torch

def get_config():
    config = config_dict.ConfigDict()

    #Experimental config
    config.transfer = False
    config.early_stopping = True
    config.max_epochs = 5000
    config.batch_size = 16
    #config.batch_size = 64
    config.result_prefix = 'results/individual_vicuna7b'

    path_dict = {
        "llama-2": "",
        "llama-2-70b": "",
        "local_unsensor": "",
        "llama-3": "",
        "Mistral": "",
        "vicuna": "",
        "llama-guard": "",
        "harmbench": ""
    }

    #Model config
    config.model_path = path_dict["llama-2"]
    config.model_name="llama-2"
    config.device = "cuda:7"
    config.dtype = torch.float16
    config.detector_path = path_dict["harmbench"]
    config.detector_device = "cuda:6"

    #SM config
    config.sm_layer = 14
    config.suffix_length = 20
    config.sample_num = 32
    config.alpha = 5

    #result
    config.generate_response = True
    config.save_record = False
    config.result_path = "../result"

    #autoDan config
    config.model_short_name = "Llama2"
    config.developer_name = "Meta"
    config.model_judge = "harmbench"
    config.model_mutate = "Mistral"
    config.judge_device = "cuda:1"
    config.mutate_device = "cuda:2"
    config.output_dir = ""
    config.output_plot_dir = ""
    config.ratio_elites = 0.1
    config.crossover = 0.5
    config.num_points = 5
    config.mutation = 0.1

    return config

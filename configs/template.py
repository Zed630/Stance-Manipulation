from ml_collections import config_dict
import torch

def get_config():
    config = config_dict.ConfigDict()

    #Experimental config
    config.early_stopping = True
    config.max_epochs = 5000
    config.batch_size = 16
    config.test_prompts_path = ""

    model_paths = {
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
    config.model_path = model_paths["llama-2"]
    config.model_name="llama-2"
    config.device = "cuda:7"
    config.dtype = torch.float16
    config.detector_path = model_paths["harmbench"]
    config.detector_device = "cuda:6"

    #SM config
    config.sm_layer = 14
    config.suffix_length = 20
    config.sample_num = 32
    config.alpha = 5

    #result
    config.generate_response = True
    config.save_record = True
    config.result_path = "../result"

    return config

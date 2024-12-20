from ml_collections import config_dict
import torch

def get_config():
    config = config_dict.ConfigDict()

    #实验参数
    config.transfer = False
    config.early_stopping = True
    config.max_epochs = 100
    config.batch_size = 16
    #config.batch_size = 64
    config.result_prefix = 'results/individual_vicuna7b'

    path_dict = {
        "llama-2": "/data/fushuangjie/llama-7b-chat/7b-chat-hf/",
        "llama-2-70b": "/data/thxpublic/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/8b17e6f4e86be78cf54afd49ddb517d4e274c13f",
        "local_unsensor": "../../llama-7b-chat/vicuna_uncensor/uncensor_model",
        "llama-3": "/data/fushuangjie/llama-7b-chat/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1",
        "Mistral": "/data/fushuangjie/llama-7b-chat/mistral-7b/mistral_model",
        "vicuna": "/data/fushuangjie/llama-7b-chat/vicuna/vicuna-7b-v1.5",
        "llama-guard": "/data/fushuangjie/llama-guard/Llama-Guard-3-8B",
        "harmbench": "/data/fushuangjie/llama-7b-chat/harmbench-llama-13b"
    }

    #模型参数
    config.model_path = path_dict["llama-2"]
    config.model_name="llama-2"
    config.device = "cuda:7"
    config.dtype = torch.float16
    config.detector_path = path_dict["harmbench"]
    config.detector_device = "cuda:6"

    #SM参数
    config.sm_layer = 14
    config.suffix_length = 20
    config.sample_num = 32
    config.alpha = 5

    #result
    config.generate_response = True
    config.save_record = False
    config.result_path = "../result"

    #autoDan参数
    config.model_short_name = "Llama2"
    config.developer_name = "Meta"
    config.model_judge = "harmbench"
    config.model_mutate = "Mistral"
    config.judge_device = "cuda:1"
    config.mutate_device = "cuda:2"
    config.output_dir = "/data/fushuangjie/whitebox/result_autodan/our_autodan_harmbench_vicuna"
    config.output_plot_dir = "/data/fushuangjie/whitebox/result_autodan/our_autodan_harmbench_vicuna/plot/"
    config.ratio_elites = 0.1
    config.crossover = 0.5
    config.num_points = 5
    config.mutation = 0.1

    return config

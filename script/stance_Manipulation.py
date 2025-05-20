from absl import flags, app
import ml_collections
from ml_collections import config_flags

import os
os.sys.path.append("..")
from tools.sm_class import LLM, Instence
from tools.util import load_prompts, save_to_jsonl, gcg_origin

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config_file', '../configs/template.py')



def main(_):
    config = FLAGS.config_file
    model = LLM(name=config.model_name, device=config.device, dtype=config.dtype, path=config.model_path)
    
    #进行gcg越狱
    instences = load_prompts(model.llm_name)
    for i in range(len(instences)):
        inst = instences[i]
        gcg_origin(model, [inst], config, case_idx=i)

if __name__ == "__main__":
    app.run(main)

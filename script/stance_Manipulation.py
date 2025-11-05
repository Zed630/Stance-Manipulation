from absl import flags, app
import ml_collections
from ml_collections import config_flags

import os
os.sys.path.append("..")
from tools.classes import LLM, Instence
from tools.util import load_prompts, save_to_jsonl, gcg_origin

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config_file', '../configs/template.py')

def main(_):
    config = FLAGS.config_file
    model = LLM(name=config.model_name, device=config.device, dtype=config.dtype, path=config.model_path)

    instences = load_prompts(config.test_prompts_path)
    for i in range(len(instences)):
        inst = instences[i]
        jailbreak(model, [inst], config, case_idx=i)

if __name__ == "__main__":
    app.run(main)

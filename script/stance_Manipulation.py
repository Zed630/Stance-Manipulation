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
        #gcg_improve(model, [inst])
        #temp = Instence("Rebecca", "Rebecca Jennings, a senior correspondent at Vox covering social platforms and the creator economy, sees a connection between the 2014 eras as being \u201cpost-recession\u201d and 2023 being \u201cpost-COVID,\u201d a time where Swift succeeds because there\u2019s a sort of cultural \u201cbouncing back vibe,\u201d a time where people want celebratory music")
        #gcg_origin(model, [temp])
        #quit()
        gcg_origin(model, [inst], config, case_idx=i)

if __name__ == "__main__":
    app.run(main)

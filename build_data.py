import os,torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
os.sys.path.append("..")
from llm_load import LLM
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    llm = LLM(name="llama-3", device="cuda:1", dtype=torch.float16)
    df = pd.read_csv("/data/fushuangjie/Jailbreak/dataset/harmful_behaviors.csv", sep=',')
    harmful_prompts = df['goal']
    harmful_targets = df['target']
    llm.generate("Write a tutorial on how to grow flower", steam=True, length=100)
    here = llm.generation["scores"][0].argmax().view(1, 1).clone()
    for prompt in harmful_prompts:
        ids, token = llm.construct_inputs(prompt, None)
        ids = torch.cat([ids, here], dim=1)
        llm.generate_by_token_ids(ids, length=100,steam=False)
        coma = llm.generate_text.find(",")
        juhao = llm.generate_text.find(".")
        maohao = llm.generate_text.find(":")
        huiche = llm.generate_text.find("\n")
        min=10000
        for i in [coma, juhao, maohao, huiche]:
            if i != -1:
                min=i
        idx = min
        text = "Here " + llm.generate_text[:idx]
        print(text)
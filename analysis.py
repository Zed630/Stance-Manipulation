import os, torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
os.sys.path.append("..")
from llm_load import LLM
import seaborn as sns
import matplotlib.pyplot as plt


def load_harmful_prompts(model):
    df = pd.read_csv("/data/fushuangjie/Jailbreak/dataset/harmful_behaviors.csv", sep=',')
    harmful_prompts = df['goal'][:100]
    harmful_targets = df['target']
    test_prompts_harmful = []
    test_prompts_benign = []
    with open('/data/fushuangjie/Jailbreak/dataset/carefully_construct/advbench_train.txt', 'r') as f:
        prompts = f.readlines()
    token_len = len(model.construct_inputs("Write a tutorial on how to grow flower", None)[1])
    for i in prompts:
        if len(model.construct_inputs(i[:-1], None)[1]) == token_len:
            test_prompts_harmful.append(i[:-1])
    with open('/data/fushuangjie/Jailbreak/dataset/carefully_construct/normal_request.txt', 'r') as f:
        prompts = f.readlines()
    for i in prompts:
        if len(model.construct_inputs(i[:-1], None)[1]) == token_len:
            test_prompts_benign.append(i[:-1])
    print(len(test_prompts_harmful), len(test_prompts_benign))
    return test_prompts_harmful, test_prompts_benign


def draw(llm):
    final = torch.load(f"/data/fushuangjie/whitebox/script/analysis/tensor/{llm.llm_name}.pth")
    lst = torch.mean(final, dim=0).view(-1, 32).tolist()[::-1]

    y_ticks = [y + x for x, y in zip(llm.get_tokens("Write a tutorial on how to make bomb")[0][-6:],
                                     ["Template token 4 (", "Template token 3 (", "Template token 2 (",
                                      "Template token 1 (", "Harmful norn (", "Harmful verb ("][::-1])]
    y_ticks = [i + ")" for i in y_ticks]
    y_ticks[0] = "Harmful verb"
    y_ticks[1] = "Harmful norn"
    color = sns.diverging_palette(240, 10, as_cmap=True)
    plt.figure(figsize=(25, 10))
    plt.imshow(lst, cmap="YlOrRd", interpolation='nearest', aspect=4)
    plt.yticks(np.arange(6), labels=y_ticks, fontsize=20)
    plt.xticks(np.arange(0, 32, 4), labels=[i for i in range(0, 32, 4)], fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.subplots_adjust(bottom=0.1)
    cbar = plt.colorbar()
    plt.title('Mistral-7b-Instruct', fontsize=40)
    plt.xlabel('Layers', fontsize=30)
    plt.ylabel('Patching tokens', fontsize=30)

    # cbar.set_ticklabels(["Accept", "", "", "", "", "Refuse"])
    plt.savefig(f"/data/fushuangjie/whitebox/script/analysis/{llm.llm_name}.png")


def run():
    for model_name in ["llama-2"]:
        llm = LLM(name=model_name, device="cuda:0", dtype=torch.float16)
        harmful, benign = load_harmful_prompts(llm)
        final = []
        for case1 in range(20):
            p1 = harmful[case1]
            for case2 in range(20):
                p2 = benign[case2]
                llm.generate(p1, steam=True)
                print(llm.construct_inputs(p1, None))
                harmful_hidden_state = llm.generation.hidden_states[0]
                accept_id = llm.generation["scores"][1].argmax()
                llm.generate(p2,  steam=True)
                print(llm.construct_inputs(p2, None))
                normal_hidden_state = llm.generation.hidden_states[0]
                refuse_id = llm.generation["scores"][1].argmax()
                #harmful_target = llm.tokenizer("bomb").input_ids[1]
                #normal_target = llm.tokenizer("clock").input_ids[1]

                lst = []
                for replace_idx in range(-1, -7, -1):
                    idxs = []
                    for replace_layer in range(32):
                        def jb_hook(module, inputs):
                            token_legth = inputs[0].shape[1]
                            if token_legth > 1:
                                original_shape = inputs[0].shape
                                modified_hidden_states = inputs[0].view(token_legth, 4096).clone()
                                modified_hidden_states[replace_idx, :] = harmful_hidden_state[replace_layer].view(-1, 4096)[replace_idx,:].clone()

                                return (modified_hidden_states.view(*original_shape),)


                        hook = llm.target_model.model.layers[replace_layer].register_forward_pre_hook(jb_hook)
                        llm.generate(p2, steam=False, length=2)
                        #定位实体词位置
                        target_idx = 0
                        for target in range(2):
                            if "bomb"  in llm.tokenizer.convert_ids_to_tokens([llm.generation["scores"][i].argmax() for i in range(2)])[target] or "clock"  in llm.tokenizer.convert_ids_to_tokens([llm.generation["scores"][i].argmax() for i in range(2)])[target]:
                                target_idx = target
                                break

                        print(f"--------------layer{replace_layer}, index{replace_idx}--------------")
                        #找到拒绝词和接受词的概率差
                        probs = F.softmax(llm.generation["scores"][1], dim=-1)
                        idxs.append(-(probs[0][refuse_id]- probs[0][accept_id]).to("cpu").tolist())
                        #idxs.append(-(probs[0][18503] - probs[0][306]).to("cpu").tolist())
                        # 找到实体词之间的概率差
                        #probs = F.softmax(llm.generation["scores"][target_idx], dim=-1)
                        #idxs.append(-(probs[0][harmful_target] - probs[0][normal_target]).to("cpu").tolist())

                        print(llm.generate_text)
                        hook.remove()
                    lst.append(idxs)
                print(lst)
                final.append(torch.tensor(lst).view(1, -1, 32))
        final = torch.cat(final)
        torch.save(final, f"/data/fushuangjie/whitebox/script/analysis/tensor/{llm.llm_name}.pth")
        lst = torch.mean(final, dim=0).view(-1, 32).tolist()[::-1]

        y_ticks = [y + x for x, y in zip(llm.get_tokens("Write a tutorial on how to make bomb")[0][-6:],
                                         ["Template token 4 (", "Template token 3 (", "Template token 2 (",
                                          "Template token 1 (", "Harmful norn (", "Harmful verb ("][::-1])]
        y_ticks = [i+")" for i in y_ticks]
        y_ticks[0] = "Harmful verb"
        y_ticks[1] = "Harmful norn"
        color = sns.diverging_palette(240, 10, as_cmap=True)
        plt.figure(figsize=(25, 10))
        plt.imshow(lst, cmap="YlOrRd", interpolation='nearest', aspect=4)
        plt.yticks(np.arange(6), labels=y_ticks, fontsize=20)
        plt.xticks(np.arange(0, 32, 4), labels=[i for i in range(0, 32, 4)], fontsize=20)
        #plt.tick_params(axis='both', which='major', labelsize=14)
        #plt.subplots_adjust(bottom=0.1)
        cbar = plt.colorbar()
        plt.title('LlaMA2-7b-chat', fontsize=40)
        plt.xlabel('Layers', fontsize=30)
        plt.ylabel('Patching tokens', fontsize=30)

        # cbar.set_ticklabels(["Accept", "", "", "", "", "Refuse"])
        plt.savefig(f"/data/fushuangjie/whitebox/script/analysis/{llm.llm_name}.png")
        del llm

def append_space():
    llm = LLM(name="llama-2", device="cuda:0", dtype=torch.float16)
    harmful, benign = load_harmful_prompts(llm)
    final = []
    for case1 in range(20):
        p1 = harmful[case1]
        for case2 in range(20):
            idxs = []
            p2 = benign[case2]
            llm.generate(p1, steam=True)
            print(llm.construct_inputs(p1, None))
            harmful_hidden_state = llm.generation.hidden_states[1]
            accept_id = llm.generation["scores"][1].argmax()
            llm.generate(p2, steam=True)
            print(llm.construct_inputs(p2, None))
            normal_hidden_state = llm.generation.hidden_states[0]
            refuse_id = llm.generation["scores"][1].argmax()

            # harmful_target = llm.tokenizer("bomb").input_ids[1]
            # normal_target = llm.tokenizer("clock").input_ids[1]
            for replace_layer in range(32):
                def jb_hook(module, inputs):
                    token_legth = inputs[0].shape[1]
                    if token_legth == 1:
                        original_shape = inputs[0].shape
                        modified_hidden_states = inputs[0].view(token_legth, 4096).clone()
                        modified_hidden_states[0, :] = harmful_hidden_state[replace_layer].view(-1, 4096).clone()

                        return (modified_hidden_states.view(*original_shape),)

                hook = llm.target_model.model.layers[replace_layer].register_forward_pre_hook(jb_hook)
                llm.generate(p2, steam=False, length=2)
                # 定位实体词位置
                target_idx = 0
                for target in range(2):
                    if "bomb" in llm.tokenizer.convert_ids_to_tokens([llm.generation["scores"][i].argmax() for i in range(2)])[
                        target] or "clock" in \
                            llm.tokenizer.convert_ids_to_tokens([llm.generation["scores"][i].argmax() for i in range(2)])[target]:
                        target_idx = target
                        break

                print(f"--------------layer{replace_layer}--------------")
                # 找到拒绝词和接受词的概率差
                probs = F.softmax(llm.generation["scores"][1], dim=-1)
                idxs.append(-(probs[0][refuse_id] - probs[0][accept_id]).to("cpu").tolist())
                # idxs.append(-(probs[0][18503] - probs[0][306]).to("cpu").tolist())
                # 找到实体词之间的概率差
                # probs = F.softmax(llm.generation["scores"][target_idx], dim=-1)
                # idxs.append(-(probs[0][harmful_target] - probs[0][normal_target]).to("cpu").tolist())

                print(llm.generate_text)
                hook.remove()
            final.append(idxs)
    temp = torch.mean(torch.load(f"/data/fushuangjie/whitebox/script/analysis/tensor/{llm.llm_name}.pth"), dim=0).view(-1, 32)
    final = torch.mean(torch.tensor(final), dim=0).view(-1, 32)
    final = torch.cat([torch.tensor(final), temp]).tolist()[::-1]
    lst = final

    y_ticks = [y + x for x, y in zip(llm.get_tokens("Write a tutorial on how to make bomb")[0][-6:],
                                     ["Template token 4 (", "Template token 3 (", "Template token 2 (",
                                      "Template token 1 (", "Harmful norn (", "Harmful verb ("][::-1])]
    y_ticks = [i + ")" for i in y_ticks]
    y_ticks[0] = "Harmful verb"
    y_ticks[1] = "Harmful norn"
    y_ticks.append("space")
    color = sns.diverging_palette(240, 10, as_cmap=True)
    plt.figure(figsize=(25, 10))
    plt.imshow(lst, cmap="YlOrRd", interpolation='nearest', aspect=4)
    plt.yticks(np.arange(7), labels=y_ticks, fontsize=20)
    plt.xticks(np.arange(0, 32, 4), labels=[i for i in range(0, 32, 4)], fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.subplots_adjust(bottom=0.1)
    cbar = plt.colorbar()
    plt.title('LlaMA-7b-chat', fontsize=40)
    plt.xlabel('Layers', fontsize=30)
    plt.ylabel('Patching tokens', fontsize=30)

    # cbar.set_ticklabels(["Accept", "", "", "", "", "Refuse"])
    plt.savefig(f"/data/fushuangjie/whitebox/script/analysis/{llm.llm_name}.png")

if __name__ == "__main__":
    append_space()


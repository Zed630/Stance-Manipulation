import torch, random, jsonlines, time, os
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from tools.sm_class import evaluator
import jsonlines

from tools.sm_class import Instence


random.seed(1)
def load_prompts(llmname="llama-2"):
    df = pd.read_csv("/data/fushuangjie/Jailbreak/dataset/harmful_behaviors.csv", sep=',')
    harmful_prompts = df['goal']
    harmful_targets = df['target']

    instences = []
    for i in range(len(harmful_prompts)):
        if llmname == "llama-3":
            instences.append(Instence(harmful_prompts[i] + ". ", harmful_targets[i].replace("Sure, h", "H")))
        else:
            instences.append(Instence(harmful_prompts[i] + " ", harmful_targets[i]))
    return instences

def save_to_jsonl(result, full_path='./result/data.jsonl'):
    """
    Saves the dataset to a JSONL file using jsonlines library.

    :param str path: The path of the file where the dataset will be saved. Defaults to 'data.jsonl'.
    """
    path, filename = os.path.split(full_path)
    if not os.path.exists(path):
        os.makedirs(path)
    with jsonlines.open(full_path, mode='a') as writer:
        writer.write(result)

def gcg_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()  # B * (L-1) * V
    shift_logits = shift_logits.transpose(1, 2)  # B * V * (L-1)
    shift_labels = labels[:, 1:].contiguous()  # B * (L-1)
    masked_loss = F.cross_entropy(shift_logits, shift_labels,
                                  reduction='none')  # B * (L-1) # CrossEntropyLoss会自动把label为-100的loss置为0

    mask = (shift_labels != -100)
    valid_elements_per_row = mask.sum(dim=1)  # B

    ans = masked_loss.sum(dim=1) / valid_elements_per_row
    assert len(ans.size()) == 1
    return ans  # B

def gcg_origin(model, instences, config, case_idx):
    suffix_length = config.suffix_length
    suffix_num = config.sample_num
    batch_size = config.batch_size
    early_stopping = config.early_stopping
    layer = config.sm_layer
    alpha = config.alpha
    eval = evaluator()

    endtoken_length = 0
    embed = model.embed
    V = embed.weight.size(0)
    aver_target = torch.load(f"./{model.llm_name}/test_{layer}.pth").to(model.device).to(model.target_model.dtype)
    origin = torch.load(f"./{model.llm_name}/test_{layer}_h.pth").to(model.device).to(model.target_model.dtype)
    direction = aver_target - origin
    # 初始化越狱后缀
    #suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor(model.tokenizer.encode("! " * suffix_length, add_special_tokens=False)[:suffix_length]).view(1, -1).to(model.device)

    #计算时间开销
    total_start = time.time()
    for inst in instences:
        ids, prompt = model.construct_inputs(inst.prompt, sys="")
        ids_temp, prompt_temp = model.construct_inputs("[[prompt]]", sys="")
        endtoken_length = max([i if prompt[-i:] == prompt_temp[-i:] else 0 for i in range(len(prompt))])
        ids = ids.to(model.device)
        prompt_len = len(prompt)
        #添加分割符号

        """if model.llm_name == "llama-3":
            ids = torch.cat([ids[:, :-model.endtoken_length], torch.tensor([256]).view(1, -1).to(model.device), ids[:, -model.endtoken_length:]], dim=1)
            prompt_len += 1"""
        inst.L1 = prompt_len - endtoken_length  # 第一个越狱字符位置
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :inst.L1], suffix_ids, ids[:, inst.L1:]], dim=1)
        inst.ids = ids
        inst.L2 = ids.size(1)
        #加上答案的ids, llama-3从0开始，llama2从1开始
        ans_ids = torch.tensor(model.tokenizer.encode(inst.target, add_special_tokens=False)[:suffix_length]).view(1, -1).to(model.device)
        refuse_ids = torch.tensor(model.tokenizer.encode(" I cannot fulfill", add_special_tokens=False)[:suffix_length]).view(1, -1).to(model.device)
        inst.whole_ids = torch.cat([ids, ans_ids], dim=1).clone()
        inst.refuse_ids = torch.cat([ids, refuse_ids], dim=1).clone()
        #labels
        labels = torch.full_like(inst.whole_ids, -100)
        labels[:, inst.L2:] = inst.whole_ids[:, inst.L2:]
        inst.labels = labels.clone()
        labels = torch.full_like(inst.refuse_ids, -100)
        labels[:, inst.L2:] = inst.refuse_ids[:, inst.L2:]
        inst.refuse_labels = labels.clone()

    # 评估和迭代
    gradient_cost = 0
    batch_cost = 0
    evaluation_cost = 0
    for i in range(config.max_epochs):
        # 构建独热矩阵输入
        totol_grad = 0

        start_time = time.time()
        for inst in instences:
            one_hot_input = F.one_hot(inst.whole_ids, num_classes=V).to(model.target_model.dtype)  # 1 * L * V
            one_hot_input.requires_grad = True
            embed_matrix = embed.weight  # V * D
            inputs_embeds = torch.matmul(one_hot_input, embed_matrix)  # 1 * L * D

            # 生成一次，找到梯度
            # gcg的loss算法
            outputs = model.target_model(inputs_embeds=inputs_embeds, labels=inst.labels, output_hidden_states=True)
            loss = outputs.loss
            #loss = outputs.loss*alpha - (outputs.hidden_states[layer][0][inst.L2-1] - inst.origin).flatten() @ direction.flatten() / direction.norm()
            #loss = - (outputs.hidden_states[20][0][inst.L2 - 1] - inst.origin).flatten() @ direction.flatten() / direction.norm()
            inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            print(loss.item(), outputs.loss)

            loss.backward()

            token_grad = one_hot_input.grad
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)  # 1 * L * V
            token_grad = -token_grad[:, inst.L1:]

            totol_grad += token_grad
        gradient_cost += time.time() - start_time

        # 选词
        with torch.no_grad():
            start_time = time.time()
            # 去掉不需要的tokens
            for token_ids in model.nonsense_ids:
                totol_grad[:, :, token_ids] = float('-inf')
            top_k_indices = torch.topk(totol_grad, dim=2, k=suffix_num).indices
            # 得到替换后的句子

            suffix_id_lst = []
            for _ in range(suffix_num):
                rel_idx = random.randint(0, suffix_length - 1)
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, suffix_num/4 - 1)]

                # 替换为suffix
                temp = suffix_ids.clone()
                temp[:, rel_idx] = new_token_id
                suffix_id_lst.append(temp)

            total_loss = 0

            for inst in instences:
                # 分batch
                if len(suffix_id_lst) > batch_size:
                    batches = [suffix_id_lst[i: i + batch_size] for i in range(0, len(suffix_id_lst), batch_size)]
                else:
                    batches = [suffix_id_lst]
                batch_loss = torch.tensor([]).to(model.device)
                for batch in batches:
                    batch_ids = []
                    for suffix in batch:
                        # 加上越狱后缀，初始化越狱token
                        ids = torch.cat([inst.ids[:, :inst.L1], suffix, inst.whole_ids[:, inst.L1+suffix_length:]], dim=1)
                        batch_ids.append(ids)

                    torch.cuda.empty_cache()
                    # 选bestone
                    # gcg_batch_loss
                    outputs = model.target_model(input_ids=torch.cat(batch_ids), output_hidden_states=True)
                    logits = outputs.logits
                    temp = gcg_loss(logits, torch.cat([inst.labels] * len(batch)))
                    #temp = gcg_loss(logits, torch.cat([inst.labels]*len(batch)))*alpha - (outputs.hidden_states[layer][:, inst.L2-1] - inst.origin).view(-1, 4096) @ direction.flatten() / direction.norm()
                    #temp = gcg_loss(logits, torch.cat([inst.labels] * len(batch)))
                    batch_loss = torch.cat([batch_loss, temp])
                total_loss += batch_loss
            best_token_ids = suffix_id_lst[torch.argmin(total_loss)]
            batch_cost += time.time() - start_time
            start_time = time.time()
            suffix_ids = best_token_ids.clone()
            _pass = 0

            for inst in instences:
                # 加上越狱后缀，初始化越狱token
                ids = torch.cat([inst.ids[:, :inst.L1], suffix_ids, inst.ids[:, inst.L1+suffix_length:]], dim=1)
                inst.whole_ids[:, inst.L1:inst.L1+suffix_length] = suffix_ids
                inst.refuse_ids[:, inst.L1:inst.L1 + suffix_length] = suffix_ids
                #是否评估并保存结果
                if config.generate_response:
                    model.generate_by_token_ids(ids, length=100)
                    if early_stopping and eval.llm_detector(inst.prompt, model.generate_text, config):
                        _pass += 1
                    whole_prompt = model.tokenizer.decode(ids[0])
                    #评估时间开销
                    evaluation_cost += time.time() - start_time
                    # 保存结果
                    data = {
                        'request': inst.prompt,
                        "prompt": ids.tolist(),
                        'iteration': i,
                        "answer": model.generate_text,
                        'eval_results': _pass > 0,
                        "full_prompt": whole_prompt,
                        'case_idx': case_idx,
                        "time_cost": time.time() - total_start,
                        "refusal_loss": model.target_model(input_ids=inst.refuse_ids, labels=inst.refuse_labels).loss.item(),
                        "loss_target": model.target_model(input_ids=inst.whole_ids, labels=inst.labels).loss.item()
                    }
                    if config.save_record:
                        save_to_jsonl(data, full_path=f'{config.result_path}/{model.llm_name}/SM_{layer}_small_sample_origin.jsonl')
                        if _pass:
                            save_to_jsonl(data, full_path=f'{config.result_path}/{model.llm_name}/SM_{layer}_small_sample_origin_successful.jsonl')
            print(f"pass:{_pass}")

        if _pass == len(instences) and early_stopping:
            break

        print(f"iteration:{i}/500")



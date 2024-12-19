import torch, random, jsonlines, time
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from gcg.evaluation import evaluator

from llm_load import Instence


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


def save_to_jsonl(result, path='data.jsonl'):
    """
    Saves the dataset to a JSONL file using jsonlines library.

    :param str path: The path of the file where the dataset will be saved. Defaults to 'data.jsonl'.
    """
    import jsonlines

    with jsonlines.open(path, mode='a') as writer:
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

def pca(x=torch.load(f"pca_x.pth").numpy()):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    x1_mean = np.mean(X_pca[:100, :], axis=0)
    x2_mean = np.mean(X_pca[100:, :], axis=0)
    return torch.tensor(pca.components_), torch.tensor(pca.mean_), torch.tensor(x1_mean), torch.tensor(x2_mean)

def gcg_improve_pca(model, instences, suffix_length=20, batch_size=128, iter_time=500, suffix_num=32, early_stopping=True, layer=14):
    component, mean, x1_mean, x2_mean = pca()

    for module in model.target_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module

    V = embed.weight.size(0)

    #初始化越狱后缀
    #suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor([1738]*suffix_length).view(1, -1).to(model.device)

    """aver_target = torch.load(f"test_{layer}.pth").to(model.device)
    origin = torch.load(f"test_{layer}_h.pth").to(model.device)
    # direction = aver_target - origin_hidden_states[14][:, -1, :]
    direction = aver_target - origin"""
    direction = (x2_mean - x1_mean).to(model.device)
    print(x2_mean)

    for inst in instences:
        prompt, ids = model.get_tokens(inst.prompt, sys="")
        inst.L1 = len(prompt) - 4  # 第一个越狱字符位置
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :-4], suffix_ids, ids[:, -4:]], dim=1)
        inst.ids = ids
        L2 = ids.size(1)
        ans_ids = torch.tensor([model.tokenizer(inst.target).input_ids]).to(model.device)
        ids = torch.cat([ids, ans_ids], dim=1)
        inst.whole_ids = ids
        labels = torch.full_like(ids, -100)
        labels[:, L2:] = ids[:, L2:]
        inst.labels = labels

    # 评估和迭代
    eval = evaluator("string")
    for i in range(iter_time):
        # 构建独热矩阵输入
        totol_grad = 0
        for inst in instences:
            one_hot_input = F.one_hot(inst.ids, num_classes=V).to(model.target_model.dtype)  # 1 * L * V
            one_hot_input.requires_grad = True
            embed_matrix = embed.weight  # V * D
            inputs_embeds = torch.matmul(one_hot_input, embed_matrix)  # 1 * L * D

            """#labels
            labels = torch.full_like(ids, -100)
            labels[:, target_slice] = ids[:, target_slice]"""

            # 生成一次，找到梯度
            # outputs = self.llm.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True, labels=labels)
            outputs = model.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            # loss = -(outputs.hidden_states[14][0][-1] - origin_hidden_states[14][:, -1, :]).flatten() @ direction.flatten() / direction.norm()
            #loss = -(outputs.hidden_states[14][0][-1] - origin).flatten() @ direction.flatten()*10 / direction.norm() + torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            #loss = -(outputs.hidden_states[layer][0][-1] - inst.origin).flatten() @ direction.flatten()  * 10 / direction.norm() + torch.norm(
            #    outputs.hidden_states[18][0][-2] - inst.semantic)
            loss = (outputs.hidden_states[layer][0][-1]-mean.to(model.device)) @ component.to(model.device).T
            #loss = -(loss - x1_mean.to(model.device)) @ direction / direction.norm()
            loss = (loss - x2_mean.to(model.device)).norm()
            print(loss)
            inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            #print(loss.item(), inst.semantic_change)
            #print((outputs.hidden_states[layer][0][-1] - loss.item()*direction-origin).norm())
            #loss = loss + (outputs.hidden_states[14][0][-1] - loss.item()*direction-origin).norm()

            loss.backward()

            token_grad = one_hot_input.grad
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)  # 1 * L * V
            token_grad = -token_grad[:, (-4-suffix_length):]

            totol_grad += token_grad

        # 选词
        with torch.no_grad():
            # 去掉不需要的tokens
            for token_ids in model.nonsense_ids:
                totol_grad[:, :, token_ids] = float('-inf')
            top_k_indices = torch.topk(totol_grad, dim=2, k=suffix_num).indices
            # 得到替换后的句子

            suffix_id_lst = []
            for _ in range(suffix_num):
                rel_idx = random.randint(0, suffix_length-1)
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, suffix_num/4-1)]

                #替换为suffix
                temp = suffix_ids.clone()
                temp[:, rel_idx] = new_token_id
                suffix_id_lst.append(temp)

            total_loss = 0


            for inst in instences:
                #分batch
                if len(suffix_id_lst)>batch_size:
                    batches = [suffix_id_lst[i: i + batch_size] for i in range(0, len(suffix_id_lst), batch_size)]
                else:
                    batches = [suffix_id_lst]
                batch_loss = torch.tensor([]).to(model.device)
                for batch in batches:
                    batch_ids = []
                    for suffix in batch:
                        # 加上越狱后缀，初始化越狱token
                        ids = torch.cat([inst.ids[:, :(-4-suffix_length)], suffix, inst.ids[:, -4:]], dim=1)
                        batch_ids.append(ids)
    
                    torch.cuda.empty_cache()
                    # 选bestone
                    outputs = model.target_model(input_ids=torch.cat(batch_ids), output_hidden_states=True)
                    #temp = -(outputs.hidden_states[layer][:, -1] - inst.origin).view(-1,
                    #                                                               4096) @ direction.flatten() / direction.norm()*10 + torch.norm(outputs.hidden_states[18][:, -2] - inst.semantic, dim=1)
                    inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)

                    temp = (outputs.hidden_states[layer][:, -1]-mean.to(model.device)).view(-1,4096) @ component.to(model.device).T
                    #temp = -(temp - x1_mean.to(model.device)) @ direction.to(model.device) / direction.norm()
                    temp = (temp - x2_mean.to(model.device)).norm(p=2, dim=1)
                    batch_loss = torch.cat([batch_loss, temp])

                total_loss += batch_loss
            best_token_ids = suffix_id_lst[torch.argmin(total_loss)]

            suffix_ids = best_token_ids.clone()
            _pass = 0
            for inst in instences:
                # 加上越狱后缀，初始化越狱token
                ids = torch.cat([inst.ids[:, :(-4-suffix_length)], suffix_ids, inst.ids[:, -4:]], dim=1)
                inst.ids = ids
                if torch.min(total_loss) < 6*len(instences):
                    model.generate_by_token_ids(ids)
                    if eval.string_evaluate(model.generate_text):
                        _pass += 1

                    whole_prompt = model.tokenizer.batch_decode(model.generation["sequences"][:, :ids.shape[1]], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
                    #保存结果
                    result = {
                        inst.prompt:[model.generate_text, i, eval.string_evaluate(model.generate_text), whole_prompt, inst.semantic_change.item()],
                    }
                    save_to_jsonl(result, path='data_1.jsonl')
            print(f"pass:{_pass}")

        if _pass == len(instences) and early_stopping:
            break

        print(f"iteration:{i}/500")


def gcg_improve_point(model, instences, suffix_length=30, batch_size=32, iter_time=500, suffix_num=32,
                    early_stopping=False, layer=10, skip=False):
    for module in model.target_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module

    V = embed.weight.size(0)
    # 初始化越狱后缀
    #suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor([1736] * suffix_length).view(1, -1).to(model.device)

    aver_target = torch.load(f"./{model.llm_name}/test_{layer}.pth").to(model.device).to(model.target_model.dtype)
    origin = torch.load(f"./{model.llm_name}/test_{layer}_h.pth").to(model.device).to(model.target_model.dtype)
    # direction = aver_target - origin_hidden_states[14][:, -1, :]
    direction = aver_target - origin

    for inst in instences:
        prompt, ids = model.get_tokens(inst.prompt, sys="")
        inst.L1 = len(prompt) - 4  # 第一个越狱字符位置
        ids = ids.to(model.device)
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        inst.origin_2 = outputs.hidden_states[layer][0][-3].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :-model.endtoken_length], suffix_ids, ids[:, -model.endtoken_length:]], dim=1)
        inst.ids = ids
        inst.min_loss = 0
        inst.refresh = 5

    # 评估和迭代
    eval = evaluator("string")
    for i in range(iter_time):
        # 构建独热矩阵输入
        totol_grad = 0
        for inst in instences:
            one_hot_input = F.one_hot(inst.ids, num_classes=V).to(model.target_model.dtype)  # 1 * L * V
            one_hot_input.requires_grad = True
            embed_matrix = embed.weight  # V * D
            inputs_embeds = torch.matmul(one_hot_input, embed_matrix)  # 1 * L * D

            # 生成一次，找到梯度
            # outputs = self.llm.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True, labels=labels)
            outputs = model.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            loss = -(outputs.hidden_states[layer][0][-1] - inst.origin).flatten() @ direction.flatten() / direction.norm()
            #loss = -(outputs.hidden_states[18][0][-1] - origin).flatten() @ direction.flatten()*10 / direction.norm() + torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            #loss = -(outputs.hidden_states[layer][0][-1] - inst.origin).flatten() @ direction.flatten()  * 10 / direction.norm()
            #loss = (outputs.hidden_states[layer][0][-1] - aver_target).norm()
            inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            print(loss.item(), inst.semantic_change)
            if loss.item()>inst.min_loss:
                inst.refresh -= 1
            else:
                inst.min_loss = loss.item()
                inst.refresh = 10
            # print((outputs.hidden_states[layer][0][-1] - loss.item()*direction-origin).norm())
            # loss = loss + (outputs.hidden_states[14][0][-1] - loss.item()*direction-origin).norm()

            loss.backward()
            token_grad = one_hot_input.grad
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)  # 1 * L * V
            token_grad = -token_grad[:, (-model.endtoken_length - suffix_length):]
            totol_grad += token_grad
        # 选词
        with torch.no_grad():
            # 去掉不需要的tokens
            for token_ids in model.nonsense_ids:
                totol_grad[:, :, token_ids] = float('-inf')
            top_k_indices = torch.topk(totol_grad, dim=2, k=suffix_num).indices
            # 得到替换后的句子

            suffix_id_lst = []
            for _ in range(suffix_num):
                rel_idx = random.randint(0, suffix_length - 1)
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, suffix_num / 4 - 1)]

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
                        ids = torch.cat([inst.ids[:, :(-model.endtoken_length - suffix_length)], suffix, inst.ids[:, -model.endtoken_length:]], dim=1)
                        batch_ids.append(ids)

                    torch.cuda.empty_cache()
                    # 选bestone
                    outputs = model.target_model(input_ids=torch.cat(batch_ids), output_hidden_states=True)
                    temp = -(outputs.hidden_states[layer][:, -1] - inst.origin).view(-1, 4096) @ direction.flatten() / direction.norm()
                    #temp = -(outputs.hidden_states[layer][:, -1] - inst.origin).view(-1,4096) @ direction.flatten() / direction.norm() *10  + torch.norm(outputs.hidden_states[18][:, -2] - inst.semantic, dim=1)
                    #temp = (outputs.hidden_states[layer][:, -1, :]-aver_target).view(-1, 4096).norm(dim=1)
                    inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
                    batch_loss = torch.cat([batch_loss, temp])
                total_loss += batch_loss
            best_token_ids = suffix_id_lst[torch.argmin(total_loss)]

            suffix_ids = best_token_ids.clone()
            _pass = 0
            for inst in instences:
                # 加上越狱后缀，初始化越狱token
                ids = torch.cat([inst.ids[:, :(-model.endtoken_length - suffix_length)], suffix_ids, inst.ids[:, -model.endtoken_length:]], dim=1)
                inst.ids = ids
                if inst.refresh <= 0 or i%20==0 or skip:
                    inst.refresh = 10
                    model.generate_by_token_ids(ids, length=300)
                    if eval.string_evaluate(model.generate_text):
                        _pass += 1
                    """whole_prompt = model.tokenizer.batch_decode(model.generation["sequences"][:, :ids.shape[1]],
                                                                skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)[0]"""
                    whole_prompt = ids.tolist()
                    # 保存结果
                    result = {
                        inst.prompt: [model.generate_text, i, eval.string_evaluate(model.generate_text), whole_prompt,
                                      inst.semantic_change.item()],
                    }
                    save_to_jsonl(result, path=f'../result/{model.llm_name}/{layer}layer_30token_500iter.jsonl')
            print(f"pass:{_pass}")

        if _pass == len(instences) and early_stopping:
            break

        print(f"iteration:{i}/500")

eval = evaluator("string")
def gcg_origin(model, instences, case_idx, suffix_length=20, batch_size=256, suffix_num=512, early_stopping=False, layer=14, alpha=5):
    for module in model.target_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module

    V = embed.weight.size(0)
    aver_target = torch.load(f"./{model.llm_name}/test_{layer}.pth").to(model.device).to(model.target_model.dtype)
    origin = torch.load(f"./{model.llm_name}/test_{layer}_h.pth").to(model.device).to(model.target_model.dtype)
    direction = aver_target - origin
    # 初始化越狱后缀
    #suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor([1738] * suffix_length).view(1, -1).to(model.device)
    #suffix_ids = torch.tensor([990] * suffix_length).view(1, -1).to(model.device)
    #计算时间开销
    total_start = time.time()
    for inst in instences:
        prompt, ids = model.get_tokens(inst.prompt, sys="")
        ids = ids.to(model.device)
        prompt_len = len(prompt)
        #添加分割符号

        """if model.llm_name == "llama-3":
            ids = torch.cat([ids[:, :-model.endtoken_length], torch.tensor([256]).view(1, -1).to(model.device), ids[:, -model.endtoken_length:]], dim=1)
            prompt_len += 1"""
        inst.L1 = prompt_len - model.endtoken_length  # 第一个越狱字符位置
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :-model.endtoken_length], suffix_ids, ids[:, -model.endtoken_length:]], dim=1)
        inst.ids = ids
        inst.L2 = ids.size(1)
        #加上答案的ids, llama-3从0开始，llama2从1开始
        if model.llm_name == "llama-3":
            ans_ids = torch.tensor([model.tokenizer(inst.target).input_ids])[:, 0:].to(model.device)
            refuse_ids = torch.tensor([model.tokenizer(" I cannot fulfill").input_ids])[:, 0:].to(model.device)
        else:
            ans_ids = torch.tensor([model.tokenizer(inst.target).input_ids])[:, 1:].to(model.device)
            refuse_ids = torch.tensor([model.tokenizer(" I cannot fulfill").input_ids])[:, 1:].to(model.device)
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
    for i in range(5000):
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
            token_grad = -token_grad[:, inst.L2 - model.endtoken_length - suffix_length:]

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
                        ids = torch.cat([inst.ids[:, :(-model.endtoken_length - suffix_length)], suffix, inst.ids[:, -model.endtoken_length:], inst.whole_ids[:, inst.L2:]], dim=1)
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
                ids = torch.cat([inst.ids[:, :(-model.endtoken_length - suffix_length)], suffix_ids, inst.ids[:, -model.endtoken_length:]], dim=1)
                inst.whole_ids[:, inst.L1:inst.L1+suffix_length] = suffix_ids
                inst.refuse_ids[:, inst.L1:inst.L1 + suffix_length] = suffix_ids
                #是否评估并保存结果
                record = True
                if record and i % 20==0:
                    model.generate_by_token_ids(ids, length=100)
                    if early_stopping and eval.llm_detector(inst.prompt, model.generate_text, device="cuda:5"):
                        _pass += 1
                    whole_prompt = model.tokenizer.decode(ids[0])
                    #评估时间开销
                    evaluation_cost += time.time() - start_time
                    # 保存结果
                    data = {
                        'request': inst.prompt,
                        "prompt": ids.tolist(),
                        'iteration': i,
                        'eval_results': False,
                        "full_prompt": whole_prompt,
                        'case_idx': case_idx,
                        "time_cost": time.time() - total_start,
                        "refusal_loss": model.target_model(input_ids=inst.refuse_ids, labels=inst.refuse_labels).loss.item(),
                        "loss_target": model.target_model(input_ids=inst.whole_ids, labels=inst.labels).loss.item()
                    }

                    #save_to_jsonl(data, path=f'../official-result/huge_sample/{model.llm_name}/SM_{layer}.jsonl')
            print(f"pass:{_pass}")

        if _pass == len(instences) and early_stopping:
            break

        print(f"iteration:{i}/500")


def gcg_improve_two_point(model, instences, suffix_length=50, batch_size=128, iter_time=3000, suffix_num=32,
                      early_stopping=True, layer=32):
    for module in model.target_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module

    V = embed.weight.size(0)
    # 初始化越狱后缀
    # suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor([1736] * suffix_length).view(1, -1).to(model.device)

    aver_target = torch.load(f"test_{layer}.pth").to(model.device)
    origin = torch.load(f"test_{layer}_h.pth").to(model.device)
    # direction = aver_target - origin_hidden_states[14][:, -1, :]
    direction = aver_target - origin
    aver_target_2 = torch.load(f"test_{layer}_3.pth").to(model.device)
    origin_2 = torch.load(f"test_{layer}_h_3.pth").to(model.device)
    # direction = aver_target - origin_hidden_states[14][:, -1, :]
    direction_2 = aver_target_2 - origin_2

    for inst in instences:
        prompt, ids = model.get_tokens(inst.prompt, sys="")
        inst.L1 = len(prompt) - 4  # 第一个越狱字符位置
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        inst.origin_2 = outputs.hidden_states[layer][0][-3].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :-4], suffix_ids, ids[:, -4:]], dim=1)
        inst.ids = ids
        inst.min_loss = 0
        inst.refresh = 5

    # 评估和迭代
    eval = evaluator("string")
    for i in range(iter_time):
        # 构建独热矩阵输入
        totol_grad = 0
        for inst in instences:
            one_hot_input = F.one_hot(inst.ids, num_classes=V).to(model.target_model.dtype)  # 1 * L * V
            one_hot_input.requires_grad = True
            embed_matrix = embed.weight  # V * D
            inputs_embeds = torch.matmul(one_hot_input, embed_matrix)  # 1 * L * D

            # 生成一次，找到梯度
            # outputs = self.llm.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True, labels=labels)
            outputs = model.target_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            loss = -(outputs.hidden_states[14][0][
                         -1] - inst.origin).flatten() @ direction.flatten() / direction.norm() - (
                               outputs.hidden_states[14][0][
                                   -3] - inst.origin_2).flatten() @ direction_2.flatten() / direction_2.norm()
            # loss = -(outputs.hidden_states[14][0][-1] - origin).flatten() @ direction.flatten()*10 / direction.norm() + torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            # loss = -(outputs.hidden_states[layer][0][-1] - inst.origin).flatten() @ direction.flatten()  * 10 / direction.norm()
            # loss = (outputs.hidden_states[layer][0][-1] - aver_target).norm()
            inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
            print(loss.item(), inst.semantic_change)
            if loss.item() > inst.min_loss:
                inst.refresh -= 1
            else:
                inst.min_loss = loss.item()
                inst.refresh = 5
            # print((outputs.hidden_states[layer][0][-1] - loss.item()*direction-origin).norm())
            # loss = loss + (outputs.hidden_states[14][0][-1] - loss.item()*direction-origin).norm()

            loss.backward()
            token_grad = one_hot_input.grad
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)  # 1 * L * V
            token_grad = -token_grad[:, (-4 - suffix_length):]
            totol_grad += token_grad
        # 选词
        with torch.no_grad():
            # 去掉不需要的tokens
            for token_ids in model.nonsense_ids:
                totol_grad[:, :, token_ids] = float('-inf')
            top_k_indices = torch.topk(totol_grad, dim=2, k=suffix_num).indices
            # 得到替换后的句子

            suffix_id_lst = []
            for _ in range(suffix_num):
                rel_idx = random.randint(0, suffix_length - 1)
                new_token_id = top_k_indices[0][rel_idx][random.randint(0, suffix_num / 4 - 1)]

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
                        ids = torch.cat([inst.ids[:, :(-4 - suffix_length)], suffix, inst.ids[:, -4:]], dim=1)
                        batch_ids.append(ids)

                    torch.cuda.empty_cache()
                    # 选bestone
                    outputs = model.target_model(input_ids=torch.cat(batch_ids), output_hidden_states=True)
                    temp = -(outputs.hidden_states[layer][:, -1] - inst.origin).view(-1,
                                                                                     4096) @ direction.flatten() / direction.norm() - (
                                       outputs.hidden_states[layer][:, -3] - inst.origin_2).view(-1,
                                                                                                 4096) @ direction_2.flatten() / direction_2.norm()
                    # temp = (outputs.hidden_states[layer][:, -1, :]-aver_target).view(-1, 4096).norm(dim=1)
                    inst.semantic_change = torch.norm(outputs.hidden_states[18][0][-2] - inst.semantic)
                    batch_loss = torch.cat([batch_loss, temp])
                total_loss += batch_loss
            best_token_ids = suffix_id_lst[torch.argmin(total_loss)]

            suffix_ids = best_token_ids.clone()
            _pass = 0
            for inst in instences:
                # 加上越狱后缀，初始化越狱token
                ids = torch.cat([inst.ids[:, :(-4 - suffix_length)], suffix_ids, inst.ids[:, -4:]], dim=1)
                inst.ids = ids
                if inst.refresh <= 0 or i % 20 == 0:
                    inst.refresh = 5
                    model.generate_by_token_ids(ids)
                    if eval.string_evaluate(model.generate_text):
                        _pass += 1
                    whole_prompt = model.tokenizer.batch_decode(model.generation["sequences"][:, :ids.shape[1]],
                                                                skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)[0]
                    # 保存结果
                    result = {
                        inst.prompt: [model.generate_text, i, eval.string_evaluate(model.generate_text), whole_prompt,
                                      inst.semantic_change.item()],
                    }
                    save_to_jsonl(result, path='data_2.jsonl')
            print(f"pass:{_pass}")

        if _pass == len(instences) and early_stopping:
            break

        print(f"iteration:{i}/500")

#loss分析
def loss_analysis(model, instences, case_idx, suffix_length=20, batch_size=160, suffix_num=256, early_stopping=False, layer=14, alpha=50):
    for module in model.target_model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module

    V = embed.weight.size(0)
    aver_target = torch.load(f"./{model.llm_name}/test_{layer}.pth").to(model.device).to(model.target_model.dtype)
    origin = torch.load(f"./{model.llm_name}/test_{layer}_h.pth").to(model.device).to(model.target_model.dtype)
    direction = aver_target - origin
    # 初始化越狱后缀
    #suffix_ids = torch.randint(0, V, (1, suffix_length)).to(model.device)
    suffix_ids = torch.tensor([1738] * suffix_length).view(1, -1).to(model.device)
    #suffix_ids = torch.tensor([990] * suffix_length).view(1, -1).to(model.device)
    #计算时间开销
    total_start = time.time()
    for inst in instences:
        prompt, ids = model.get_tokens(" " + inst.prompt[:-1], sys="")
        ids = ids.to(model.device)
        prompt_len = len(prompt)
        #添加分割符号

        """if model.llm_name == "llama-3":
            ids = torch.cat([ids[:, :-model.endtoken_length], torch.tensor([256]).view(1, -1).to(model.device), ids[:, -model.endtoken_length:]], dim=1)
            prompt_len += 1"""
        inst.L1 = prompt_len - model.endtoken_length  # 第一个越狱字符位置
        outputs = model.target_model(input_ids=ids.view(1, -1), output_hidden_states=True)
        inst.origin = outputs.hidden_states[layer][0][-1].detach().clone()
        inst.semantic = outputs.hidden_states[18][0][-2].detach().clone()
        # 加上越狱后缀，初始化越狱token
        ids = torch.cat([ids[:, :model.endtoken_length], suffix_ids, ids[:, model.endtoken_length:]], dim=1)
        inst.ids = ids
        inst.L2 = ids.size(1)
        #加上答案的ids, llama-3从0开始，llama2从1开始
        if model.llm_name == "llama-3":
            ans_ids = torch.tensor([model.tokenizer(inst.target).input_ids])[:, 0:].to(model.device)
        else:
            ans_ids = torch.tensor([model.tokenizer(inst.target).input_ids])[:, 1:].to(model.device)
        ids = torch.cat([ids, ans_ids], dim=1)
        inst.whole_ids = ids
        #labels
        labels = torch.full_like(ids, -100)
        labels[:, inst.L2:] = ids[:, inst.L2:]
        inst.labels = labels

    # 评估和迭代
    gradient_cost = 0
    batch_cost = 0
    evaluation_cost = 0
    best_loss = 10000
    for i in range(5):
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
            inst.last_loss = loss.item()
            print(loss.item(), outputs.loss)

            loss.backward()
            token_grad = one_hot_input.grad
            token_grad = token_grad / (token_grad.norm(dim=2, keepdim=True) + 1e-6)  # 1 * L * V
            token_grad = -token_grad[:, inst.L2 - model.endtoken_length - suffix_length:]
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
            for _ in range(32):
                suffix_id_lst = []
                for rel_idx in range(20):
                    for top_idx in range(8):
                        new_token_id = top_k_indices[0][rel_idx][top_idx+_*8]
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
                            ids = torch.cat([inst.ids[:, :(model.endtoken_length)], suffix, inst.ids[:, (model.endtoken_length+ suffix_length):], inst.whole_ids[:, inst.L2:]], dim=1)
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
                print(torch.min(total_loss), best_loss)
                decloss = ((total_loss - inst.last_loss) < 0).sum().item()
                if torch.min(total_loss) < best_loss:
                    suffix_ids = best_token_ids.clone()
                    min_loc = (_+1)*8
                    best_loss = torch.min(total_loss)
                print(decloss, min_loc, "-----------------------")
                _pass = 0
                #分析loss分布
            for inst in instences:
                inst.last_loss = best_loss


            for inst in instences:
                # 加上越狱后缀，初始化越狱token
                ids = torch.cat([inst.ids[:, :(model.endtoken_length)], suffix_ids, inst.ids[:, (model.endtoken_length+ suffix_length):]], dim=1)
                inst.whole_ids[:, model.endtoken_length:model.endtoken_length+suffix_length] = suffix_ids
                model.generate_by_token_ids(ids, length=100)
                if early_stopping and eval.llm_detector(inst.prompt, model.generate_text, device="cuda:0"):
                    _pass += 1
                whole_prompt = model.tokenizer.decode(ids[0])
                # 评估时间开销
                evaluation_cost += time.time() - start_time
                # 保存结果
                data = {
                    'request': inst.prompt,
                    "prompt": ids.tolist(),
                    'iteration': i,
                    'eval_results': _pass==1,
                    "full_prompt": whole_prompt,
                    'case_idx': case_idx,
                    "time_cost": time.time() - total_start
                }

                #save_to_jsonl(data, path=f'../official-result/huge_sample/{model.llm_name}/SM_{layer}_test.jsonl')

        if _pass == len(instences) and early_stopping:
            break
        print(f"iteration:{i}/500")

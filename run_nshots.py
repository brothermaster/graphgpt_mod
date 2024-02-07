import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(instruct_item, graph_data_all): 
    """
    instruct_item: 某一条 指令数据，样例如下
        {"id": "addtasks_test_4", "graph":{"node_idx": 4, "edge_index": [[],[]], "node_list": [4, 1, 2, 0, 3]},"conversations": [{"from": "human", "value":""},{"from": "gpt", "value":""}]}
    graph_data_all:所有图数据 样例如下 {dataset_name:pyg_data}
    """
    # 图字典
    graph_dict = instruct_item['graph']
    # 边
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    # 点
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    # 目标点
    target_node = copy.deepcopy(graph_dict['node_idx'])
    # 数据集名称
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    # 节点特征
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    # 节点个数
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
    # PyG 子图
    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    res = {'graph_data': graph_ret, 'graph_token_len': cur_token_len}
    return res


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    # prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []

    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(idx_list) == num_gpus: 
        idx_list.append(len(prompt_file))
    elif len(idx_list) == num_gpus + 1: 
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx],start_idx
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file,start_idx=0):
    # Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print('start loading')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    # 添加 图token
    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    graph_tower = model.get_model().graph_tower
    graph_tower.to(device='cuda', dtype=torch.bfloat16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    
    graph_data_all = torch.load(args.graph_data_path)
    res_data = []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # 读取 该节点图数据  {'graph_data': PyG图, 'graph_token_len': 节点个数}
        graph_dict = load_graph(instruct_item, graph_data_all)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']
        # 人类问题
        qs = instruct_item["conversations"][0]["value"]
        # 替换人类问题中的 图token
        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)
        # 对话格式
        conv_mode = "v1"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        # 应用模板到 输入文本
        conv = conv_templates[args.conv_mode].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # 转换输入为张量
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        # ？？？
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = "</s>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data.graph_node = graph_data.graph_node.to(torch.bfloat16)
        # graph_data.edge_index = graph_data.edge_index.to(torch.bfloat16)
        # 推理
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data.cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])
        # 获取输入长度
        input_token_len = input_ids.shape[1]
        # 验证输出前半部分与 输入是否一致
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # 获取输出
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
    with open(osp.join(args.output_res_path, f'{args.prompting_file}_{start_idx}.json'), "w") as fout:
        json.dump(res_data, fout)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)
    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    args = parser.parse_args()

    prompt_file = load_prompting_file(args.prompting_file)
    eval_model(args,prompt_file)

    # ray.init()
    # run_eval(args, args.num_gpus)


# protobuf             4.22.3
import json
import random
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as th
from torch_geometric.utils import subgraph
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.utils import from_scipy_sparse_matrix
import logging
import copy

def get_logger(fname,): 
    # 1. 获取logger对象,这是日志记录的入口
    logger = logging.getLogger('process logging')

    # 2. 设置日志级别 
    logger.setLevel(logging.INFO)

    # 3. 创建日志文件handler
    log_file = f'./log_dir/{fname}.log'
    file_handler = logging.FileHandler(log_file)

    # 4. 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter) 

    # 5. 将handler添加到logger
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler() 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 6. 记录日志  
    # logger.info('App started') 
    return logger
logger =  get_logger('process_train_arxiv')
# 设置全局种子
th.manual_seed(0)
random.seed(123)  # 设置随机数种子，这里使用了整数123作为种子
# 数据集名称
dsname ='addtasks'
stage = "pretrain" # zeroshots
tasks = ['income','car']
# 文件路径
edge_path = ''
vertice_path = ''
seed_path = ''
train_nodes_path = ''
val_nodes_path = ''
test_nodes_path = ''

edge = pd.DataFrame(np.array(
    [[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,5],
     [1,2,3,4,0,2,3,4,0,1,3,4,1,0,2,4,5]]).T,columns = ['src','dst'])
vertice = pd.DataFrame(
    np.concatenate([
        np.arange(7).reshape(7,1),
        np.random.rand(7,109)],axis=1))

seed_df = pd.DataFrame(
    [
     [0,123,"car",'Is the vertice has a car?',"a:1;b:2",'yes',1],
     [1,122,"car",'Is the vertice has a car?',"a:1;b:2",'yes',1],

     [3,125,"car",'Is the vertice has a car?',"a:1;b:2",'yes',1],
     [4,116,"car",'Is the vertice has a car?',"a:1;b:2",'yes',1],
     [5,128,"car",'Is the vertice has a car?',"a:1;b:2",'yes',1],

     [0,123,"income",'Is income of the vertice high?',"a:1;b:2",'yes',1],
     [1,122,"income",'Is income of the vertice high?',"a:1;b:2",'yes',1],
     [2,121,"income",'Is income of the vertice high?',"a:1;b:2",'yes',1],
     [3,125,"income",'Is income of the vertice high?',"a:1;b:2",'yes',1],
     [4,116,"income",'Is income of the vertice high?',"a:1;b:2",'yes',1],

    ],
     columns = ['id','iden_num','task','text','context','label_text','label']
)
# 读取点，边，seed表
# edge = pd.read_pickle(edge_path) # columns = ['src','dst','desc']
edge = edge[['src','dst']]
# vertice = pd.read_pickle(vertice_path) # 
# seed_df = pd.read_pickle(seed_path) # columns = ['id','iden_num','task','text','label_text','label']
seed_df = seed_df[['id','iden_num','task','text','context','label_text','label']]
seed_df.columns = ['id','iden_num','task','instruction','input','output','label']

# 读取训练，验证，测试点
# train_nodes = pd.read_pickle(train_nodes_path) # columns = ['id']
# val_nodes = pd.read_pickle(val_nodes_path)   # columns = ['id']
# test_nodes = pd.read_pickle(test_nodes_path)  # columns = ['id']
train_nodes = pd.DataFrame([0,1])
val_nodes = pd.DataFrame([2,3])
test_nodes = pd.DataFrame([4,5])
train_nodes.columns = ['node_idx']
train_nodes['split_type'] = 'train'
val_nodes.columns = ['node_idx']
val_nodes['split_type'] = 'val'
test_nodes.columns = ['node_idx']
test_nodes['split_type'] = 'test'
res_df = pd.concat([train_nodes, val_nodes, test_nodes])
res_df = res_df[['split_type','node_idx']]

'''
instruct dataset: 
[{'id': 'dsname_train_nodeidx', 'graph': [edge_row, edge_col], 'conversations': [{'from': 'human', 'value': 'human prompting.\n<graph>'}, {'from': 'gpt', 'value': 'gpt response'}]}, {...}]

graph_token: <graph>
'''
edge_index = th.from_numpy(edge[['src','dst']].values.T)
x = th.from_numpy(vertice.values[:,1:]).float()
pyg_data = Data(x = x, edge_index = edge_index, edge_attr = None, num_nodes = len(vertice))
# Data(num_nodes=169343, x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243], train_mask=[169343], val_mask=[169343], test_mask=[169343], edge_index=[169343, 169343, nnz=2315598])
graph_dict = {dsname:pyg_data}
th.save(graph_dict,f'./instruct_ds/{dsname}/graph.pt')

cat_ds = {}
# 遍历任务 使id唯一
for task,group_t in seed_df.groupby('task'):
    if task in tasks:
        pass
    else:
        continue
    instruct_list = {}
    instruct_list.update(zip(group_t['id'].tolist(), group_t[['instruction','input','output']].to_dict(orient='records')))

    sp_dict = {} # {"train":[],"val":[],"test":[]}
    # 遍历 分割
    res_df_gp = res_df.groupby('split_type')
    for name,group in res_df_gp:
        select_idx = group['node_idx'].tolist()
        split_type = name

        # 指令集合
        instruct_ds = []
        for nidx in tqdm(select_idx): 
            center_node = nidx 
            num_hops = 2
            num_neighbors = 10

            # 邻居采样    
            sampler = NeighborLoader(pyg_data, input_nodes=th.Tensor([center_node]).long(),
                                    num_neighbors=[num_neighbors] * num_hops, 
                                    batch_size=1)

            # 获取子图    
            sampled_data = next(iter(sampler))
            # for sampled_data in sampler:

            try:
                temp_dict = {}
                temp_dict['id'] = f'{dsname}_{split_type}_{nidx}'
                temp_dict['graph'] = {'node_idx':nidx, 'edge_index': sampled_data.edge_index.tolist(), 'node_list': sampled_data.n_id.tolist()}
                conv_list = []
                conv_temp = {}
                conv_temp['from'] = 'human'
                conv_temp['value'] = 'Given a transction graph: \n<graph>\nwhere the 0th node is the target person, with the following information: \n' \
                                        + "Context:" + instruct_list[nidx]['input'] + "\n" \
                                        + "Question:"+ instruct_list[nidx]['instruction']
                conv_list.append(copy.deepcopy(conv_temp))

                conv_temp['from'] = 'gpt'
                conv_temp['value'] = instruct_list[nidx]['output']
                conv_list.append(copy.deepcopy(conv_temp))

                temp_dict['conversations'] = conv_list

                instruct_ds.append(temp_dict)
            except Exception as e:
                logger.info(e)

        if not os.path.exists(f"./instruct_ds/{dsname}"):
            os.makedirs(f"./instruct_ds/{dsname}")
        #                         数据集名称/阶段/任务/分割
        with open(f'./instruct_ds/{dsname}/{stage}_{task}_{split_type}.json', 'w') as f:
            json.dump(instruct_ds, f)

    #     sp_dict[split_type] = instruct_ds
    # cat_ds[task] = sp_dict

# for i in ['train','val','test']:
#     print(i)
#     tmp_ds = []
#     for j in tasks:
#         tmp_ds + cat_ds[j][i]

#     with open(f'./instruct_ds/{dsname}_{stage}_{i}.json', 'w') as f:
#         json.dump(instruct_ds, f)

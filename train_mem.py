# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from graphgpt.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

# 修改 compute_metric 为acc ，（使用示例中的函数进行尝试）
# 答：已修改

# 确认验证集上的指标如何计算，一步计算？生成式计算？（逐步debug找到相关位置）
# 答：一步计算

# 测试传入服务器的docker镜像是否可用

# 修改zero-shots 和 few-shots 部分代码

# graph transformer 输入特征维度与 当前数据不同，如何修改？
# 答：不是用预训练对齐的GNN，使用未训练的GNN，预训练阶段 同时调整 GNN + GNN_PROJ。而zero-shots 和 few-shots 仅调整GNN_PROJ

from graphgpt.train.train_graph import train

if __name__ == "__main__":
    train()



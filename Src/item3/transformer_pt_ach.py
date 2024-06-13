from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from torch import nn
from transformers import AutoConfig
import torch
from math import sqrt
import torch.nn.functional as F

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"
show(model, "bert", tokenizer, text, display_mode="light", layer=0, head=8)

# 先对文本进行词元化，句子中的每个词元都被映射到词元分析器的词表中的唯一ID
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(f"the inputs ids: {inputs.input_ids}")

# 使用AutoConfig类加载了与bert-base-uncased checkpoint相关联的config.json文件
# 每个输入的ID将映射到nn.Embedding中存储的30522个嵌入向量之一，其中每个向量维度为768.
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(f"the token_emb is: {token_emb}")

# 通过输入ID，可以生成嵌入向量
inputs_embeds = token_emb(inputs.input_ids)
print(f"the inputs embeds size: {inputs_embeds.size()}")

# 使用点积作为相似度函数来计算注意力分数
query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
print(f"the socres size is: {scores.size()}")

weights = F.softmax(scores, dim=-1)
print(f"the weights is: {weights.sum(dim=1)}")

attn_outputs = torch.bmm(weights, value)
print(f"attn_output shape is: {attn_outputs.shape}")


# 把以上步骤封装成一个函数
def scaled_dot_product_attention(query, key, value):
  dim_k = query.size(-1)
  scores = torch.bmm(query, key.transpose(1, 2) / sqrt(dim_k))

  weights = F.softmax(scores, dim=-1)
  return torch.bmm(weights, value)

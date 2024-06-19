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


# 多头注意力
# 我们将query、key和value视为相等来计算注意力分数和权重。
# 首先编写一个单独的注意力头的类：
class AttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)

  def forward(self, hidden_state):
    attn_outputs = scaled_dot_product_attention(
      self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
    )
    return attn_outputs


# 初始化三个独立的线性层，对于嵌入向量执行矩阵乘法，以生成形状为[batch_size, seq_len, head_dim]张量
# 其中head_dim是我们要投影的维数数量。


# 完整的多注意力层
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
      [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, hidden_state):
    x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x

# 前馈层
class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
  
  def forward(self, x):
    x = self.linear_1(x)
    x = self.gelu(x)
    x = self.linear_2(x)
    x = self.dropout(x)
    return x

class TransformerEncoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttention(config)
    self.feed_forward = FeedForward(config)
  
  def forward(self, x):
    hidden_state = self.layer_norm_1(x)
    x = x +self.attention(hidden_state)
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x
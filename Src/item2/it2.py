from datasets import list_datasets, load_dataset
import huggingface_hub
import pandas as pd

# all datatsets 查看
# all_datasets = huggingface_hub.list_datasets()
# all_datasets = list_datasets()

# print(f"There are {len(all_datasets)} datsets current available on the hub")
# print(f"The 10 are: {all_datasets[:10]}")

# 返回一个Dataset类实例
# 类似python的列表
# 行为：len(Dataset), Dataset[0]，

emotions = load_dataset("emotion")
print(emotions)

train_ds = emotions["train"]
print(train_ds)

train_ds_len = len(train_ds)
train_simple_i = train_ds[0]
train_ds_col= train_ds.column_names

print(train_ds_len)
print(train_simple_i)
print(train_ds_col)

print(train_ds.features)

# 使用切片查看几行数据
print(train_ds[:5])

# 
print(train_ds["text"][:5])

# Datasets 到 DataFrame
print(">>>>>>>>")
emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())
print(">>>>>>>>")
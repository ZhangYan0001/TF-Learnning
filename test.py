from multiprocessing import context
from transformers import pipeline
import pandas as pd
from transformers.pipelines import AggregationStrategy

text = ""
with open("./text.txt", "r") as f:
  text = f.readlines()

text = "".join(text)
print()
print("OUTPUT>>>>", text)
print()

# 使用情绪分析流水线
classifier = pipeline("sentiment-analysis")
output = classifier(
  "We are very happy to introduce pipeline to the transformers repository."
)
# print(pd.DataFrame(output))
print("OUTPUT>>>>\n", output)


# 命名实体识别

ner_tagger = pipeline("ner", aggregation_strategy="simple")
ner_output = ner_tagger(
  "Tom and Jerry all know some birds are not meant to be caged, their feathers are just too bright. In China, have a WeChat"
)
print("OUTPUT>>>>\n", ner_output)
print(pd.DataFrame(ner_output))

# 问答

reader = pipeline("question-answering")
question = "What does the customer want?"
reader_output = reader(question=question, context=text)
print("\n", "OUTPUT>>>>\n", pd.DataFrame([reader_output]))

# 文本摘要

summarizer = pipeline("summarization")
summarizer_output = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print("\n", "OUTPUT>>>>\n", summarizer_output[0]["summary_text"])


# 翻译
translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh")
translator_output = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print("\n", "OUTPUT>>>>\n", translator_output[0]["translation_text"])

# 文本生成
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
generator_output = generator(prompt, max_length=200)
print("\n", "OUTPUT>>>>\n", generator_output[0]["generated_text"], "\n")


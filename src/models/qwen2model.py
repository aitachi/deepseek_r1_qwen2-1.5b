from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 确保模型目录存在
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
local_dir = os.path.join(model_dir, "qwen2-1.5b-instruct")

# 确保模型目录存在
os.makedirs(local_dir, exist_ok=True)

print(f"模型将保存到: {local_dir}")

# 指定模型名称
model_name = "Qwen/Qwen2-1.5B-Instruct"

# 下载 tokenizer 并保存到本地
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)

# 下载模型并保存到本地
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_dir)

print(f"模型已成功下载并存储到: {local_dir}")
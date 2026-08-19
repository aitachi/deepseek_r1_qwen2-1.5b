import os

# 设置环境变量以解决OpenMP冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"  # 禁用 wandb

import torch
import psutil
import GPUtil
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_model_path = os.path.join(project_root, "models", "qwen2-1.5b-instruct")
output_path = os.path.join(project_root, "models", "cold_start_completed")  # 冷启动适配器输出路径

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

print(f"基础模型路径: {base_model_path}")
print(f"冷启动输出路径: {output_path}")

# 验证CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("可用的GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# 创建简单的资源监控函数
def print_resource_usage():
    # CPU使用率
    cpu_percent = psutil.cpu_percent()
    # 内存使用
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)

    # GPU使用情况
    gpu_info = ""
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info += f"\nGPU {i}: {gpu.name}"
                gpu_info += f"\n  - 显存占用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
                gpu_info += f"\n  - GPU利用率: {gpu.load * 100:.1f}%"
        except:
            gpu_info = "\nGPU信息获取失败"
    else:
        gpu_info = "\n未检测到GPU"

    print(f"系统资源使用情况:")
    print(f"CPU使用率: {cpu_percent}%")
    print(f"内存使用: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory.percent}%)")
    print(gpu_info)
    print("-" * 50)


print("正在加载模型和tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 确保tokenizer有正确的padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("已设置padding token为EOS token")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("已清理GPU缓存")

# 加载模型
print("使用8位量化加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=True,  # 使用8位量化减少显存占用
    device_map="auto",
    torch_dtype=torch.float16
)

# 准备模型进行8位训练
print("为8位训练准备基础模型...")
model = prepare_model_for_kbit_training(model)

# 配置LoRA适配器
print("配置LoRA适配器...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA注意力维度
    lora_alpha=32,  # LoRA alpha参数
    lora_dropout=0.1,  # dropout概率
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 目标模块
)

# 获取PEFT模型
model = get_peft_model(model, peft_config)
print("PEFT模型准备完毕")

# 创建训练数据集用于冷启动，包含标准优秀的逻辑推理问答对
samples = [
    # 基础问答对
    {"input": "你好", "output": "你好！我是AI助手，有什么可以帮助你的？"},
    {"input": "什么是人工智能？", "output": "人工智能(AI)是计算机科学的一个分支，旨在开发能够模拟人类智能行为的系统。"},

    # 高质量逻辑推理示例1
    {
        "input": "一个水箱以2立方米/分钟的速度注水，同时以1立方米/分钟的速度漏水。如果水箱的容积是20立方米，且初始为空，需要多少分钟才能装满水箱？",
        "output": "我来逐步分析这个问题：\n\n1. 水箱的注水速度是2立方米/分钟\n2. 水箱的漏水速度是1立方米/分钟\n3. 因此，净注水速度 = 2 - 1 = 1立方米/分钟\n4. 水箱总容积是20立方米\n5. 计算填满时间：总容积 ÷ 净注水速度 = 20 ÷ 1 = 20分钟\n\n所以，需要20分钟才能装满水箱。"},

    # 高质量逻辑推理示例2
    {
        "input": "在一个房间里有三盏灯，对应的三个开关在房间外面。你可以随意操作开关，但只能进入房间一次。如何确定每个开关对应哪盏灯？",
        "output": "我来分析这个问题并给出解决方案：\n\n1. 首先分析问题：我们需要找出三个开关和三盏灯之间的对应关系，但只能进入房间一次\n2. 因为灯有热效应，所以可以利用这一点\n3. 具体操作步骤：\n   a. 打开第一个开关，等待10分钟\n   b. 关闭第一个开关，同时打开第二个开关\n   c. 立即进入房间\n4. 进入房间后，通过以下方式判断：\n   - 亮着的灯对应第二个开关\n   - 熄灭但摸上去很热的灯对应第一个开关\n   - 熄灭且冷的灯对应第三个开关\n\n这种方法利用了灯的物理特性（发热）来获取额外信息，从而在只进入房间一次的情况下确定所有开关的对应关系。"},

    # 数学推理示例
    {"input": "已知等比数列{an}的前3项为2, 6, 18，求该数列的第10项。",
     "output": "解决这个问题需要找出等比数列的公比，然后计算第10项。\n\n1. 首先，已知前3项分别是a₁ = 2, a₂ = 6, a₃ = 18\n2. 等比数列的公比q = a₂/a₁ = 6/2 = 3\n3. 验证：a₃ = a₂ × q = 6 × 3 = 18，确认无误\n4. 等比数列的通项公式是aₙ = a₁ × q^(n-1)\n5. 代入数据：a₁₀ = 2 × 3^(10-1) = 2 × 3^9 = 2 × 19683 = 39366\n\n因此，该等比数列的第10项是39366。"},

    # 逻辑推理示例
    {"input": "如果所有的A都是B，所有的B都是C，那么以下哪个陈述一定为真？",
     "output": "让我通过逻辑推理分析这个问题：\n\n已知条件：\n- 所有的A都是B\n- 所有的B都是C\n\n从这两个前提可以得出的逻辑结论：\n\n1. 所有的A都是C（正确）\n   - 因为所有A都是B，而所有B都是C，所以所有A必然都是C\n   - 这是三段论推理的有效形式\n\n2. 所有的C都是B（不一定）\n   - 我们只知道B是C的子集，但不知道C中是否还有其他非B的元素\n   - 因此这个陈述不能确定为真\n\n3. 所有的C都是A（不一定）\n   - 同样的，A是B的子集，B是C的子集，但我们不能确定A和C是否相等\n   - 这个陈述也不能确定为真\n\n因此，唯一一定为真的陈述是：所有的A都是C。"}
]


# 准备为训练格式化的数据
def format_instruction(sample):
    return f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"


formatted_samples = [format_instruction(sample) for sample in samples]


# 创建数据集
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)  # 增加长度以适应复杂推理


train_dataset = Dataset.from_dict({"text": formatted_samples})
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,  # 增加轮数以更好地学习逻辑推理
    learning_rate=2e-4,
    fp16=False,
    save_strategy="epoch",
    logging_steps=1,
    save_total_limit=1,
    report_to=None,
    optim="adamw_torch",
    seed=42
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# 打印资源使用情况
print("\n====== 冷启动训练前资源状态 ======")
print_resource_usage()

# 开始训练
print("开始冷启动训练...")
trainer.train()

# 保存模型
print(f"保存LoRA适配器到 {output_path}")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("冷启动训练完成")

# 训练后资源状态
print("\n====== 冷启动训练后资源状态 ======")
print_resource_usage()

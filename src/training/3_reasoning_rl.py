import os

# 设置环境变量以解决OpenMP冲突和CUDA断言
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"  # 禁用 wandb
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 帮助调试CUDA错误
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # 启用设备端断言

import torch
import psutil
import GPUtil
import time
import threading
import traceback
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
from datetime import timedelta
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
import gc

# 设置模型路径
# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_model_path = os.path.join(project_root, "models", "qwen2-1.5b-instruct")
adapter_path = os.path.join(project_root, "models", "cold_start_completed")
output_path = os.path.join(project_root, "checkpoints", "rejection_sft_model")  # 输出路径

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

print(f"基础模型路径: {base_model_path}")
print(f"适配器路径: {adapter_path}")
print(f"输出路径: {output_path}")

# 确保冷启动适配器存在，如果不存在则提示用户
if not os.path.exists(adapter_path):
    print(f"警告: 冷启动适配器 '{adapter_path}' 不存在")
    print("请先运行冷启动训练: python src/training/cold_start.py")
    print("或者设置适配器路径为None以使用原始模型")

    # 询问用户是否继续
    user_input = input("是否继续而不使用适配器？(y/n): ")
    if user_input.lower() != 'y':
        print("退出程序。请先运行冷启动训练。")
        exit(1)
    else:
        print("将继续而不使用适配器...")
        adapter_path = None

# 验证CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("可用的GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# 资源监控函数
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


# 定期监控资源使用的线程
def resource_monitor_thread(interval=60, stop_event=None):
    """每隔指定时间监控资源使用情况"""
    while not stop_event.is_set():
        print_resource_usage()
        time.sleep(interval)


print("正在加载tokenizer...")
# 首先只加载tokenizer以节省内存
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 确保tokenizer有正确的padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("已设置padding token为EOS token")

# 先清理GPU缓存和Python垃圾收集
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("已清理GPU缓存")

gc.collect()
print("已执行Python垃圾收集")


# 自定义评估准确性函数
def evaluate_accuracy(prompt, response):
    """简单的评估函数，用于判断回答是否准确"""
    try:
        # 这是一个简化版本 - 实际应用中需要实现更复杂的评估逻辑
        if "导数" in prompt.lower() or "derivative" in prompt.lower():
            keywords = ["导数", "derivative", "f'(x)", "微分", "differential"]
            return 0.8 if any(kw in response.lower() for kw in keywords) else 0.3

        elif "方程" in prompt.lower() or "equation" in prompt.lower():
            keywords = ["方程", "equation", "x =", "y =", "解", "solution"]
            return 0.8 if any(kw in response.lower() for kw in keywords) else 0.3

        elif "证明" in prompt.lower() or "prove" in prompt.lower():
            keywords = ["证明", "prove", "假设", "assume", "因此", "therefore", "所以", "QED"]
            return 0.8 if any(kw in response.lower() for kw in keywords) else 0.3

        # 默认返回值
        return 0.5
    except Exception as e:
        print(f"评估准确性时出错: {e}")
        return 0.5  # 出错时返回中等分数


# 检查响应是否有效（没有混合语言等）的函数
def is_valid_response(response):
    """检查生成的回复是否有效"""
    try:
        # 检查响应长度
        if len(response) < 10:  # 太短的响应可能不是有效回答
            return False

        # 检查混合语言（简化版）
        english_ratio = len(re.findall(r'[a-zA-Z]', response)) / max(len(response), 1)
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', response)) / max(len(response), 1)

        # 对中文或英文回复有不同的标准
        if chinese_ratio > 0.3:  # 主要是中文回复
            if english_ratio > 0.4:  # 但包含太多英文
                return False
        elif english_ratio > 0.3:  # 主要是英文回复
            if chinese_ratio > 0.4:  # 但包含太多中文
                return False
        else:  # 既不是主要中文也不是主要英文
            return False

        # 检查非常长的段落
        if max(len(p) for p in response.split("\n\n")) > 800:
            return False

        # 检查过多的代码块
        code_blocks = len(re.findall(r'```', response))
        if code_blocks > 8:  # 超过4个代码块
            return False

        return True
    except Exception as e:
        print(f"检查响应有效性时出错: {e}")
        return False  # 出错时返回无效


# 安全加载模型的函数
def safe_load_model(model_path, tokenizer, adapter_path=None, attempt=1, max_attempts=3):
    """安全地加载模型，处理可能的错误"""
    if attempt > max_attempts:
        raise RuntimeError(f"在{max_attempts}次尝试后无法加载模型")

    try:
        print(f"尝试 {attempt}/{max_attempts} 加载模型...")
        # 使用CPU加载基础模型以避免一些CUDA问题
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 让模型决定如何分配内存
            low_cpu_mem_usage=True
        )

        if adapter_path:
            print(f"尝试加载适配器 {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)

        # 将模型移至GPU
        if torch.cuda.is_available():
            # 不使用.to(device)，而是通过device_map让模型自行决定
            pass

        return model
    except Exception as e:
        print(f"加载模型失败 (尝试 {attempt}/{max_attempts}): {e}")
        # 清理GPU缓存和进行垃圾收集
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)  # 等待一段时间
        return safe_load_model(model_path, tokenizer, adapter_path, attempt + 1, max_attempts)


# 安全生成文本的函数
def safe_generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, attempt=1, max_attempts=3):
    """安全地生成文本，处理可能的CUDA错误"""
    if attempt > max_attempts:
        print(f"在{max_attempts}次尝试后无法生成文本")
        return "无法生成回复。"

    try:
        # 使用模型的输入格式
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # 尝试使用更稳定的生成设置
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,  # 禁用波束搜索以减少内存使用
            repetition_penalty=1.2,  # 添加重复惩罚
            no_repeat_ngram_size=3  # 防止3-gram重复
        )

        # 解码输出
        full_response = tokenizer.decode(output[0], skip_special_tokens=False)

        # 尝试提取助手的回复部分
        try:
            response = full_response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        except:
            response = full_response

        return response
    except Exception as e:
        print(f"生成文本失败 (尝试 {attempt}/{max_attempts}): {e}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)  # 等待一段时间
        return safe_generate(model, tokenizer, prompt, max_new_tokens, temperature, attempt + 1, max_attempts)


# 使用拒绝采样生成响应的函数
def generate_with_rejection(model, tokenizer, prompt, num_samples=2, max_retries=5):
    """使用拒绝采样方法生成高质量回复"""
    valid_responses = []

    # 尝试生成多个样本并筛选
    for retry in range(max_retries):
        if len(valid_responses) >= num_samples:
            break

        print(f"生成第 {retry + 1}/{max_retries} 次尝试...")

        try:
            # 安全生成回复
            response = safe_generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=512,
                temperature=0.7
            )

            print(f"生成的回复长度: {len(response)}")

            if is_valid_response(response):
                # 检查响应是否正确（简化版）
                accuracy = evaluate_accuracy(prompt, response)
                print(f"回复评分: {accuracy}")
                if accuracy > 0.5:
                    valid_responses.append(response)
                    print("✓ 接受回复")
                else:
                    print("✗ 拒绝回复 (低评分)")
            else:
                print("✗ 拒绝回复 (无效格式)")

        except Exception as e:
            print(f"生成过程中出错: {str(e)}")

    # 如果没有有效响应，返回一个默认回复
    if not valid_responses:
        print("! 未找到有效回复，使用默认回复")
        return [f"我将逐步解决这个问题: {prompt}"]

    return valid_responses


print("开始加载模型...")
# 使用安全加载函数
try:
    model = safe_load_model(base_model_path, tokenizer, adapter_path)
    print("成功加载模型和适配器")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("尝试仅加载基础模型...")
    try:
        model = safe_load_model(base_model_path, tokenizer)
        print("成功加载基础模型")
    except Exception as e:
        print(f"加载基础模型也失败: {e}")
        raise RuntimeError("无法加载任何模型，无法继续训练")

# 确认GPU可用性
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("警告: 未检测到GPU，将使用CPU运行")
    device = torch.device("cpu")

# 打印模型所在设备
print(f"模型加载到设备: {model.device}")

# 推理和非推理提示的示例（减少样本数量以节省时间）
reasoning_prompts = [
    "什么是函数 f(x) = x^3 + 2x^2 - 5x + 7 的导数？",
    "解方程组：2x + y = 5 和 3x - 2y = 4",
    "证明任意三角形的三个内角和为180度。"
]

non_reasoning_prompts = [
    "写一个关于机器人学习绘画的短故事。",
    "解释机器学习和深度学习之间的区别。"
]

# 为SFT生成数据
sft_data = []

# 处理推理提示
print("\n====== 开始为推理任务生成数据 ======")
for idx, prompt in enumerate(reasoning_prompts):
    print(f"\n处理推理提示 [{idx + 1}/{len(reasoning_prompts)}]: {prompt}")
    responses = generate_with_rejection(model, tokenizer, prompt)
    if responses:
        response = responses[0]  # 采用第一个有效响应

        # 随机添加思维链前缀（80%概率）
        if random.random() < 0.8:
            print("为推理任务添加思维链...")
            # 生成思维链
            cot_prompt = f"我需要解决这个问题: {prompt}。请一步一步地思考。"
            try:
                cot = safe_generate(
                    model,
                    tokenizer,
                    cot_prompt,
                    max_new_tokens=256,
                    temperature=0.7
                )

                # 使用特殊标记结构格式化
                response = f"|special_token|{cot}|special_token|{response}"
                print("已添加思维链")
            except Exception as e:
                print(f"添加思维链时出错: {e}")

        sft_data.append({
            "prompt": prompt,
            "response": response
        })
        print(f"已添加推理任务回复，最终长度: {len(response)}")

# 处理非推理提示
print("\n====== 开始为非推理任务生成数据 ======")
for idx, prompt in enumerate(non_reasoning_prompts):
    print(f"\n处理非推理提示 [{idx + 1}/{len(non_reasoning_prompts)}]: {prompt}")

    try:
        response = safe_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=512,
            temperature=0.7
        )

        # 对于某些非推理任务，我们可能想添加简单的思维链
        if random.random() < 0.3:  # 30%的概率添加思维链
            print("为非推理任务添加思维链...")
            # 简化的思维链生成
            cot_prompt = f"请一步一步思考如何回答这个问题: {prompt}"
            try:
                cot = safe_generate(
                    model,
                    tokenizer,
                    cot_prompt,
                    max_new_tokens=256,
                    temperature=0.7
                )

                # 使用特殊标记结构格式化
                response = f"|special_token|{cot}|special_token|{response}"
                print("已添加思维链")
            except Exception as e:
                print(f"添加思维链时出错: {e}")

        sft_data.append({
            "prompt": prompt,
            "response": response
        })
        print(f"已添加非推理任务回复，长度: {len(response)}")
    except Exception as e:
        print(f"处理非推理提示时出错: {e}")

# 显示SFT数据集示例
print("\n====== SFT数据集示例 ======")
for i, example in enumerate(sft_data[:min(2, len(sft_data))]):
    print(f"\n示例 {i + 1}:")
    print(f"提示: {example['prompt']}")
    print(f"回复 (前200字符): {example['response'][:200]}...")

# 检查数据集是否为空
if not sft_data:
    print("错误: SFT数据集为空，无法继续训练")
    raise RuntimeError("SFT数据集为空")


# 准备用于微调的数据集
def preprocess_function(examples, max_length=512):
    """预处理函数，带有错误处理"""
    try:
        batch_inputs = []

        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            response = examples["response"][i]
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            batch_inputs.append(text)

        tokenized_inputs = tokenizer(
            batch_inputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 对于因果语言模型，标签与输入ID相同
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        return tokenized_inputs
    except Exception as e:
        print(f"预处理数据时出错: {e}")
        # 创建一个最小的有效输入
        tokenized_inputs = tokenizer(
            ["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs


# 转换为HuggingFace数据集
print("\n创建训练数据集...")
dataset = Dataset.from_dict({
    "prompt": [item["prompt"] for item in sft_data],
    "response": [item["response"] for item in sft_data]
})

# 应用预处理函数
print("处理训练数据...")
try:
    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, max_length=768),  # 增加最大长度以适应思维链
        batched=True,
        remove_columns=dataset.column_names
    )
    print(f"数据集大小: {len(processed_dataset)}")
except Exception as e:
    print(f"处理数据集时出错: {e}")
    print("创建备用最小数据集...")
    # 创建一个最小的数据集以便训练能够继续
    minimal_data = [
        {"prompt": "测试问题", "response": "测试回答"}
    ]
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in minimal_data],
        "response": [item["response"] for item in minimal_data]
    })
    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, max_length=64),
        batched=True,
        remove_columns=dataset.column_names
    )

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("已清理GPU缓存")

# 设置训练参数
print("配置训练参数...")
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 增加梯度累积步数
    num_train_epochs=2,  # 减少训练轮数以避免过拟合
    learning_rate=2e-5,  # 调整学习率
    warmup_steps=50,  # 添加预热步骤
    max_grad_norm=0.3,  # 降低梯度裁剪阈值
    fp16=False,  # 禁用FP16
    bf16=False,  # 禁用BF16
    save_steps=20,  # 更频繁保存
    eval_steps=None,  # 禁用评估
    logging_steps=1,  # 设为1确保每步都有输出
    save_total_limit=2,
    report_to=[],  # 禁用所有报告工具，包括 wandb
    optim="adamw_torch",
    dataloader_pin_memory=False,  # 减少内存使用
    remove_unused_columns=True,  # 移除未使用的列
    seed=42,
    debug=False,  # 禁用debug模式，可能导致其他问题
    # 降低内存使用的选项
    gradient_checkpointing=True,  # 启用梯度检查点
    ddp_find_unused_parameters=False,
    dataloader_num_workers=0,  # 减少数据加载工作线程
)

# 创建trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,  # 确保传入tokenizer
)

# 任务计时器和资源监控
print("\n====== 开始拒绝采样SFT训练 ======")
print_resource_usage()  # 训练前的资源使用情况

# 创建停止标志和监控线程
stop_monitor = threading.Event()
monitor_thread = threading.Thread(
    target=resource_monitor_thread,
    args=(60, stop_monitor)  # 每60秒打印一次资源状态
)
monitor_thread.daemon = True  # 守护线程，主线程结束时自动结束

start_time = time.time()


# 定义回调函数来监控训练进度
class TrainingProgressCallback(TrainerCallback):
    def __init__(self):
        self.step = 0
        self.last_log_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_log_time >= 10:  # 每10秒至少打印一次
            print(f"\n开始训练步骤 {self.step + 1}...")
            self.last_log_time = current_time

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        print(f"完成训练步骤 {self.step}")

        # 检查是否长时间没有进度
        elapsed = time.time() - start_time
        if elapsed > 600 and self.step == 0:  # 如果10分钟内没有完成一个步骤
            print("警告: 训练似乎卡住了。请检查GPU资源和模型配置。")
            print_resource_usage()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志时清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 添加回调
trainer.add_callback(TrainingProgressCallback())

# 开始训练
try:
    # 启动监控线程
    monitor_thread.start()
    print("已启动资源监控线程（每60秒更新）")

    # 设置训练超时机制（30分钟）
    training_timeout = 30 * 60
    print(f"设置训练超时时间为{training_timeout // 60}分钟")

    # 使用超时机制运行训练
    start_train_time = time.time()
    trainer.train()

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {timedelta(seconds=int(total_time))}")

    # 保存微调后的模型
    print(f"保存适配器到 {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("拒绝采样SFT训练完成并保存模型")

except Exception as e:
    print(f"训练过程中发生错误: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()  # 打印详细的堆栈跟踪

    # 尝试保存当前模型状态
    try:
        emergency_save_path = f"{output_path}_emergency_save"
        print(f"尝试将模型紧急保存到 {emergency_save_path}")
        model.save_pretrained(emergency_save_path)
        tokenizer.save_pretrained(emergency_save_path)
        print("已成功保存模型应急备份")
    except Exception as save_error:
        print(f"保存应急备份时出错: {save_error}")

    # 检查是否是显存不足的问题
    if "CUDA out of memory" in str(e):
        print("\n诊断: GPU显存不足。请尝试以下解决方案:")
        print("1. 使用更小的模型")
        print("2. 进一步减小生成的token数量")
        print("3. 减少批处理大小")
        print("4. 增加梯度累积步数")

    # 检查是否是数值不稳定的问题
    elif "nan" in str(e).lower() or "inf" in str(e).lower():
        print("\n诊断: 训练数值不稳定。请尝试以下解决方案:")
        print("1. 进一步降低学习率")
        print("2. 增加梯度裁剪强度 (max_grad_norm=0.1)")

    # CUDA断言错误
    elif "CUDA error: device-side assert triggered" in str(e):
        print("\n诊断: CUDA设备断言错误。可能的解决方案:")
        print("1. 使用CPU模式训练 (设置device_map='cpu')")
        print("2. 减小批量大小和序列长度")
        print("3. 检查输入数据是否有无效标记")
        print("4. 尝试禁用量化 (不使用8位或4位量化)")

finally:
    # 停止监控线程
    stop_monitor.set()
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=1.0)

    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
    gc.collect()
    print("已执行Python垃圾收集")

    # 训练结束时的资源使用情况
    print("\n训练结束时的系统资源状态:")
    print_resource_usage()

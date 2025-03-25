# QWEN2-1.5B Full SFT 训练日志分析报告
基于2025-03-18训练日志的全面分析

## 摘要
本报告分析了QWEN2-1.5B模型的SFT(Supervised Fine-Tuning)训练过程日志。训练从2025年3月18日开始，使用了一个包含120万+样本的数据集，在单块RTX 4070 GPU上进行。训练初期表现正常，损失值逐渐下降，但在后期训练中出现了"NaN"值问题，表明训练过程遇到了数值稳定性问题。

## 训练参数与环境
![image](https://github.com/user-attachments/assets/87bc7eb6-b33c-4dd7-bd36-094819339ece)

## 硬件环境
![image](https://github.com/user-attachments/assets/21b6a654-67da-44c1-abcc-3438a9939459)

## 训练进度分析
### 损失变化趋势
![image](https://github.com/user-attachments/assets/ebd5e833-b65e-492d-a61e-6c187f4bd56c)
![image](https://github.com/user-attachments/assets/fe7b7496-afef-4a3e-8204-d7e027407fe3)
![image](https://github.com/user-attachments/assets/962f54fd-89b1-4e9a-93bc-ae01ad4ba445)
![image](https://github.com/user-attachments/assets/a1e21ed7-c944-4cc7-8d24-380d310e551a)

## 训练过程关键发现
### 1. 初期训练状态
![image](https://github.com/user-attachments/assets/d462eb86-513b-4fee-8e20-55ed8105497a)

### 2. 训练中期评估
由于日志中缺少中期完整数据，但可以看出模型在前300步内，平均损失从4.040下降到2.613，表现出良好的学习趋势。每个批次的学习时间估计也从初始的3238分钟逐渐调整到更合理的797分钟左右。

### 3. 训练后期问题
![image](https://github.com/user-attachments/assets/36a683f9-1127-42cc-b075-fcf8a097c9e8)

### 4. 资源使用情况变化
![image](https://github.com/user-attachments/assets/fe43b644-b887-472f-8d72-f56f588b304e)

### 5. 检查点保存情况
![image](https://github.com/user-attachments/assets/6953af24-95af-478b-93ad-df259d66fac4)

## 结论与建议
![image](https://github.com/user-attachments/assets/bdfb42cd-27b3-4d66-9adb-daad245b188e)




   








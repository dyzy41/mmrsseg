import json
import matplotlib.pyplot as plt


# 从JSON文件中读取数据
data = []
with open('work_dirs/mmseg_text_whub/20231029_034045/vis_data/20231029_034045.json') as file:
    for line in file:
        entry = json.loads(line)
        data.append(entry)

# 提取lr和loss数据
steps = [entry['step'] for entry in data]
lr = [entry['lr'] for entry in data]
loss = [entry['loss'] for entry in data]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(steps, lr, label='Learning Rate', marker='o')
plt.plot(steps, loss, label='Loss', marker='x')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Learning Rate and Loss Visualization')
plt.legend()
plt.grid(True)

# 显示图表
plt.savefig('loss_lr.png')

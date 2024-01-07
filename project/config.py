import os
from dotenv import find_dotenv


# Configurations
class CFG:
    project_name = os.path.dirname(find_dotenv())
    data_dir = os.path.join(project_name, 'input', 'data5404')
    train_data_dir = os.path.join(data_dir, '训练集数据')
    test_A_data_dir = os.path.join(data_dir, 'A榜测试集数据')
    labels = ["A", "B", "C", "D", "E"]
    label2id = {val: idx for idx, val in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}
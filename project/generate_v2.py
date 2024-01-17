import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import CFG


# Load data
label_data = pd.read_csv(os.path.join(CFG.train_data_dir, '文件标签汇总数据.csv'))
train_csv_folder = os.path.join(CFG.train_data_dir, 'csv文件')
test_A_csv_folder = os.path.join(CFG.test_A_data_dir, 'csv文件')
train_image_folder = os.path.join(CFG.project_name, 'project/image/训练集数据')
test_A_image_folder = os.path.join(CFG.project_name, 'project/image/A榜测试集数据')


def csv2png(csv_folder, image_folder, mode='train'):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    for index, csv_file in tqdm(enumerate(os.listdir(csv_folder))):
        # 读取CSV文件
        csv_path = os.path.join(csv_folder, csv_file)
        data = pd.read_csv(csv_path)
        if mode == 'train':
            label = label_data[label_data['fileName']
                               == csv_file]['defectType'].values[0]
            img_path = os.path.join(
                image_folder, label, f"{csv_file.replace(csv_file[-4:], '.png')}")
            if not os.path.exists(os.path.join(image_folder, label)):
                os.makedirs(os.path.join(image_folder, label))
        else:
            img_path = os.path.join(
                image_folder, csv_file.replace(csv_file[-4:], '.png'))
        if os.path.exists(img_path):
            continue
        length_x = data['X'].max() - data['X'].min()
        length_y = data['Y'].max() - data['Y'].min()
        image_array = np.zeros((length_x + 1, length_y + 1))
        min_x = data['X'].min()
        min_y = data['Y'].min()
        value_mean = data['Value'].mean()
        value_std = data['Value'].std()
        for _, row in data.iterrows():
            x, y, value = int(
                row['X'] - min_x), int(row['Y'] - min_y), row['Value']
            image_array[x, y] = (value - value_mean) / value_std
        # 将数组转换为图像
        image = Image.fromarray(np.uint8(image_array * 255))  # 转换为0-255范围的图像
        image = image.resize((224, 224))
        image.save(img_path)
    print(f"Finished converting {mode} data!")


csv2png(train_csv_folder, train_image_folder)
csv2png(test_A_csv_folder, test_A_image_folder, 'test')

import os
import random
import cv2
import torchvision.transforms as transforms
from PIL import Image

label_dict = ['cloudy', 'haze', 'rainy', 'snow', 'sunny', 'thunder']  # decode by list index
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_input_path(input_path):
    '''
    检车路径下的所有文件
    :param input_path: path
    :param jsonlist: json文件列表
    :param imagelist: img文件列表
    :return:
    '''
    jsonlist = []
    imagelist = []
    Csv_list = []

    def get_file_path(root_path, file_list):
        '''
        获取跟文件下的所有文件 包括子文件夹中的文件保存于
        file——list中

        :param root_path: 需要获取文件的根目录
        :param file_list: 保存文件列表
        :return: 空
        '''
        dir_or_files = os.listdir(root_path)
        for dir_file in dir_or_files:
            dir_file_path = os.path.join(root_path, dir_file)
            if os.path.isdir(dir_file_path):
                #                 pass
                #                 pass
                get_file_path(dir_file_path, file_list)
            else:
                file_list.append(dir_file_path)

    file_list = []
    get_file_path(input_path, file_list)

    for i in file_list:
        d_lss = os.path.split(i)[-1].split('.')[-1]
        if d_lss == 'json':
            jsonlist.append(i)
        elif d_lss == 'csv' or d_lss == 'xlsx':
            Csv_list.append(i)
        elif d_lss == 'jpg' or d_lss == 'png':
            imagelist.append(i)
    return jsonlist, imagelist, Csv_list


class WeatherDataSet:
    def __init__(self, input_path, size, tarin, transform):
        _, imagelist, _ = get_input_path(input_path)
        # imagelist=imagelist[:100]
        random.shuffle(imagelist)
        self.size = size
        cut_point = int(0.8 * len(imagelist))

        if tarin:
            self.files = imagelist[:cut_point]
        else:
            self.files = imagelist[cut_point:]

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = cv2.imread(self.files[item])
        img = cv2.resize(img, self.size, cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = None
        for index, tar in enumerate(label_dict):
            if tar in self.files[item]:
                label = index
        img = Image.fromarray(img)  # 这里ndarray_image为原来的numpy数组类型的输入

        return self.transform(img), label


if __name__ == '__main__':
    input_path = r'D:\WorkSpace\DataSet\weather_classification'
    dataset = WeatherDataSet(input_path, (512, 512))
    for img, label in dataset:
        print(label)
    # print(len(dataset))

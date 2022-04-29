import numpy as np
import abc


class Evalute_base:
    '''
    评估类
        设计子类
            目标价测评估
            语义分割评估
    '''

    @abc.abstractmethod
    def __init__(self, GT_json_file, Pred_json_file, class_list):
        self.class_list = class_list

        self.confusion_matrix = self.build_confusion_matrix(GT_json_file, Pred_json_file)

    @abc.abstractmethod
    def build_confusion_matrix(self, GT_json_file, Pred_json_file):
        '''
        构造混淆矩阵 抽象函数
        需要 子类根据 进行计算
            例如
                2d的目标搜索算法
                2d的分割
        :param GT_json_file:标注真值文件
        :param Pred_json_file:预测结果文件
        :return:混淆矩阵
        '''
        raise NotImplementedError

    def get_precision(self):
        '''
        通过混淆矩阵获取精确率
        :return:
        '''
        ret = []
        for index, label in enumerate(self.class_list):
            # ret[str(label)] = self.confusion_matrix[index, index] / self.confusion_matrix[:, index]
            ret.append(self.confusion_matrix[index, index] / np.sum(self.confusion_matrix[:, index]))
        return ret

    def get_recall(self):
        '''
        通过混淆矩阵获取召回率
        :return:
        '''
        # ret = {}
        ret = []
        for index, label in enumerate(self.class_list):
            # ret[str(label)] = self.confusion_matrix[index, index] / self.confusion_matrix[index, :]
            ret.append(self.confusion_matrix[index, index] / np.sum(self.confusion_matrix[index, :]))

        return ret

    def get_acc(self):
        '''
        通过混淆矩阵获取准确率
        :return:
        '''
        ret = 0
        for index, label in enumerate(self.class_list):
            ret += self.confusion_matrix[index, index]
        return ret / np.sum(self.confusion_matrix)

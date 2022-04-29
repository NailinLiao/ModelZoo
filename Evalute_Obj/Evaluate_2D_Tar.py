import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from Analysis import Analysis

from Evalute_Obj.Evaluate_Base import Evalute_base


class Evalute_target(Evalute_base):
    def __init__(self, GT_json, Pred_json, class_list):
        '''
        获取所有的目标
        遍历目标计算所有iou
        构造混淆矩阵
        :param GT_json:
        :param Pred_json:
        '''
        self.threshold = 0.5
        super().__init__(GT_json, Pred_json, class_list)

    def build_confusion_matrix(self, GT_json_file, Pred_json_file):
        # iou>the & True    TP
        # iou>the & False   误检
        # iou<the & True   背景误检
        # None iou         漏检
        matrix = np.zeros((len(self.class_list) + 1, len(self.class_list) + 1))
        GT_Boxs = Analysis.yolo_2d_tar(GT_json_file)
        Pre_Boxs = Analysis.yolo_2d_tar(Pred_json_file)
        iou_matrix = np.zeros((len(GT_Boxs), len(Pre_Boxs)))
        GT_label = []
        PD_label = []
        for index_PD, PD in enumerate(Pre_Boxs):
            pre_label = PD['label']
            PD_label.append(pre_label)

        for index_GT, GT in enumerate(GT_Boxs):
            gt_label = GT['label']
            gt_box = GT['box']
            GT_label.append(gt_label)
            for index_PD, PD in enumerate(Pre_Boxs):
                pre_box = PD['box']
                iou = self.compute_iou(gt_box, pre_box)
                iou_matrix[index_GT, index_PD] = iou
        for GT_index, row in enumerate(iou_matrix):
            max_iou_prd_index = np.unravel_index(np.argmax(row), row.shape)
            if row[max_iou_prd_index] > self.threshold and GT_label[GT_index] == PD_label[max_iou_prd_index[0]]:
                class_index = self.class_list.index(GT_label[GT_index])
                class_PD = self.class_list.index(PD_label[max_iou_prd_index[0]])

                matrix[class_index, class_PD] += 1
            elif row[max_iou_prd_index] > self.threshold and GT_label[GT_index] != PD_label[max_iou_prd_index[0]]:
                class_GT = self.class_list.index(GT_label[GT_index])
                class_PD = self.class_list.index(PD_label[max_iou_prd_index[0]])
                matrix[class_GT, class_PD] += 1
            elif row[max_iou_prd_index] < self.threshold and GT_label[GT_index] == PD_label[max_iou_prd_index[0]]:
                class_PD = self.class_list.index(PD_label[max_iou_prd_index[0]])
                matrix[0, class_PD] += 1
            elif len(row[row > 0]) == 0:
                class_GT = self.class_list.index(GT_label[GT_index])
                matrix[class_GT, 0] += 1
        self.iou_matrix = iou_matrix
        self.GT_label = GT_label
        self.PD_label = PD_label
        return matrix

    def compute_iou(self, rec1, rec2):
        """
        computing IoU [1,0,3,2]
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def get_class_iou(self):
        ret = {}
        for i in self.class_list:
            ret[i] = []
        for PD_index, PD in enumerate(self.PD_label):
            for GT_index, GT in enumerate(self.GT_label):
                tar_iou = self.iou_matrix[GT_index, PD_index]
                if tar_iou > self.threshold and GT == PD:
                    ret[GT].append(tar_iou)
        for key in ret:
            ret[key] = np.mean(ret[key])
        return ret


if __name__ == '__main__':
    Pred = r'E:\WorkSpace\Evaluate_POJO\Test_data\Tar\Pre\ILSVRC2012_test_00000018.xml'
    GT = r'E:\WorkSpace\Evaluate_POJO\Test_data\Tar\True\ILSVRC2012_test_00000018.xml'
    class_list = ['Dog', 'Person']
    tar_class = Evalute_target(GT, Pred, class_list)
    print(tar_class.get_class_iou())

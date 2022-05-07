# point cloud
from Evalute_Obj.Evaluate_Base import Evalute_base
from Analysis import Analysis
import numpy as np


class Evalute_3d_tar(Evalute_base):

    def __init__(self, GT_file, Pred_file, class_list):
        self.threshold = 0.5
        super().__init__(GT_file, Pred_file, class_list)

    def compute_3d_iou(self, box1, box2):
        '''
               box [x1,y1,z1,x2,y2,z2]   分别是两对角定点的坐标
           '''
        area1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
        area2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
        area_sum = area1 + area2

        # 计算重叠部分 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        z1 = max(box1[2], box2[2])
        x2 = min(box1[3], box2[3])
        y2 = min(box1[4], box2[4])
        z2 = min(box1[5], box2[5])
        if x1 >= x2 or y1 >= y2 or z1 >= z2:
            return 0
        else:
            inter_area = (x2 - x1) * (y2 - y1) * (z2 - z1)

        return inter_area / (area_sum - inter_area)


    def build_confusion_matrix(self, GT_file, Pred_file):
        # iou>the & True    TP
        # iou>the & False   误检
        # iou<the & True   背景误检
        # None iou         漏检
        matrix = np.zeros((len(self.class_list) + 1, len(self.class_list) + 1))
        GT_Boxs = Analysis.yolo_3d_tar(GT_file)
        Pre_Boxs = Analysis.yolo_3d_tar(Pred_file)
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
                iou = self.compute_3d_iou(gt_box, pre_box)
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
    Pred = r'E:\WorkSpace\Evaluate_POJO\Test_data\3d_box\Pre\0000000010.pcd.txt'
    GT = r'E:\WorkSpace\Evaluate_POJO\Test_data\3d_box\True\0000000010.pcd.txt'
    class_list = ['dontCare']
    tar_class = Evalute_3d_seg(GT, Pred, class_list)
    print(tar_class.get_class_iou())

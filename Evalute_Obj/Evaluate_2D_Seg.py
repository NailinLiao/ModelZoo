from Evalute_Obj.Evaluate_Base import Evalute_base
import json
import numpy as np
import cv2
from Analysis import Analysis


class Evalute_seg(Evalute_base):

    def __init__(self, GT_json, Pred_json, class_list):
        '''
        构造两个mask
        拉平
        比较
        构造混淆矩阵
        :param GT_json:
        :param Pred_json:
        '''
        # super(Evalute_seg, self).__init__(GT_json, Pred_json, class_dict)
        # Evalute_base.__init__(self, GT_json, Pred_json, class_dict)
        super().__init__(GT_json, Pred_json, class_list)

    def build_confusion_matrix(self, GT_json_file, Pred_json_file):
        n_class = len(self.class_list) + 1
        label_true = self.build_mask(GT_json_file)
        label_pred = self.build_mask(Pred_json_file)

        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def build_mask(self, json_file):

        tar = Analysis.labelme_2d_seg(json_file)

        h, w = tar['img_size']
        cnts = tar['cnts']
        mask = np.zeros((h, w))
        for cnt in cnts:
            cv2.fillPoly(mask, [np.array(cnt['points'], dtype='int')], (self.class_list.index(cnt['label']) + 1))
        return np.array(mask, dtype='int64')

    def get_dice(self):
        # dice = 2TP / (2TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix,
                                                               axis=0)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = 2 * intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def get_pix_iou(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU


if __name__ == '__main__':
    Pred = r'E:\WorkSpace\Evaluate_POJO\Test_data\Seg\Pre\ILSVRC2012_test_00000004.json'
    GT = r'E:\WorkSpace\Evaluate_POJO\Test_data\Seg\True\ILSVRC2012_test_00000004.json'
    class_list = ['Dog', 'Person']
    seg_class = Evalute_seg(GT, Pred, class_list)
    print(seg_class.confusion_matrix)
    print(seg_class.get_dice())
    print(seg_class.get_pix_iou())
    print(seg_class.get_recall())
    print(seg_class.get_precision())

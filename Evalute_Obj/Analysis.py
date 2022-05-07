import json
import xml.etree.ElementTree as ET
import numpy as np


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


class Analysis():
    @staticmethod
    def labelme_2d_seg(json_file):
        '''
        :param json_file: json file path
        :return: {'img_size': (366, 400),
                    'cnts': [
                                {'label': 'Dog',
                                'points': [[145.03816793893128, 120.69465648854961],
                                            .......
                                          [171.3740458015267, 118.78625954198472],
                                          [160.68702290076334, 115.73282442748092]],
                                'group_id': None,
                                'shape_type': 'polygon',
                                'flags': {}}
                            ]
                    }
        '''

        json_data = json.load(open(json_file, 'r', encoding="utf-8"))
        h, w = json_data['imageHeight'], json_data['imageWidth']
        cnts = json_data['shapes']

        return {
            'img_size': (h, w),
            'cnts': cnts
        }

    @staticmethod
    def yolo_2d_tar(xml_file):
        '''
        :param xml_file:
        :return:{
                    'label': label,
                    'box': [h1,x1,h2,x2],  # Warning order
                }
        '''
        tree = ET.parse(xml_file)
        root = tree.getroot()
        ret = []
        for child in root:
            if child.tag == 'object':
                label = ''
                box = []
                for i in child:

                    if i.tag == 'name':
                        label = i.text
                    if i.tag == 'bndbox':
                        for a in i:
                            box.append(int(a.text))
                box = swapPositions(box, 0, 1)
                box = swapPositions(box, 2, 3)
                ret.append({
                    'label': label,
                    'box': box,
                })
        return ret

    @staticmethod
    def yolo_3d_tar(txt_file):

        def transition_point(centre, size):
            centre = list(map(float, centre))
            size = list(map(float, size))
            centre = np.array(centre)
            size = np.array(size) / 2

            A = np.array(centre - size).tolist()
            B = np.array(centre + size).tolist()
            A += B
            return A

            # box[x1, y1, z1, x2, y2, z2]

        # 分别是两对角定点的坐标
        ret = []
        file_handle = open(txt_file, mode='r')
        contents = file_handle.readlines()
        for line in contents:
            split_data = str(line).split(' ')
            box = transition_point(split_data[1:4], split_data[4:7])
            ret.append({
                'label': split_data[0],
                'box': box,
            })
        return ret


if __name__ == '__main__':
    txt_file = r'E:\WorkSpace\Evaluate_POJO\Test_data\3d_box\Pre\0000000010.pcd.txt'
    boxs = Analysis.yolo_3d_tar(txt_file)
    for box in boxs:
        print(box)

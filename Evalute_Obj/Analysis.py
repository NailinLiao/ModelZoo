import json
import xml.etree.ElementTree as ET


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

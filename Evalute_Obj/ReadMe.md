### 该模块为验证模块
针对已经产生的结果与GT比较生成指标数据
- 实现
    - 2D目标搜索评估
    - 2D分割评估
- 指标项目
    - 准确率
    - 精确率
    - 召回率
    - IOU(类别)
    - DICE

### 使用说明

- Analysis
    - 目标文件解析
    - labelme_2d_seg将labelme 5.0格式转换为


- 2D 分割接收的格式

```python
{'img_size': (366, 400),
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
```

- 2D 目标搜索

```python
{
    'label': label,
    'box': [h1, x1, h2, x2],  # Warning order
}
```

- Evaluate_2D_Seg.py
    - 实现2d分割评估
- Evaluate_2D_Tar.py
    - 实现2d目标搜索评估


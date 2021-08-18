from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class VisDroneDataset(CocoDataset):
    CLASSES = ( 'pedestrian',
                'people',
                'bicycle',
                'car',
                'van',
                'truck',
                'tricycle',
                'awning-tricycle',
                'bus',
                'motor',)
    # CLASSES = ( 'pedestrian',
    #             'people',
    #             'bicycle',
    #             'car',
    #             'van',
    #             'truck',
    #             'tricycle',
    #             'awning-tricycle',
    #             'bus',
    #             'motor',
    #             'ren',
    #             'lianglun',
    #             'silun',
    #             'kache',
    #             'sanlun')
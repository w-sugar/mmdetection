from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class VIRATDataset(CocoDataset):
    CLASSES = ('person', 
               'somepersons', 
               'pull_or_push')

    
from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class MEVADataset(CocoDataset):
    # CLASSES = ('person', 
    #            'somepersons', 
    #            'pull_or_push')
    # CLASSES = ('persononly',)
    # CLASSES = ('persononly', 'personobject')
    CLASSES = ('person',)
    # CLASSES = ('personobject',)
    
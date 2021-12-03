from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class VIRATDataset(CocoDataset):
    # CLASSES = ('person', 
    #            'somepersons', 
    #            'pull_or_push')
    # CLASSES = ('persononly',)
    # CLASSES = ('persononly', 'personobject')
    CLASSES = ('personperson',)
    # CLASSES = ('personobject',)

@DATASETS.register_module()
class VIRATDataset2(CocoDataset):
    # CLASSES = ('person', 
    #            'somepersons', 
    #            'pull_or_push')
    # CLASSES = ('persononly',)
    CLASSES = ('persononly', 'personobject')
    # CLASSES = ('personperson',)
    # CLASSES = ('personobject',)

    
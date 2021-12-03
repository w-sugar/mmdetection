bash tools/dist_train.sh configs/virat/virat_cascade_persononly\&personobject.py 4 --work-dir=/data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_persononly\&personobject/

bash tools/dist_train.sh configs/virat/virat_cascade_personobject.py 4 --work-dir=/data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_personobject/

bash tools/dist_test.sh configs/virat/virat_cascade_persononly\&personobject.py /data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_persononly\&personobject/latest.pth 4 --format-only --eval-options jsonfile_prefix=/data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_persononly\&personobject/cascade_personandobject_val_allframes_32step16

bash tools/dist_test.sh configs/virat/virat_cascade_personobject.py /data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_personobject/latest.pth 4 --format-only --eval-options jsonfile_prefix=/data/sugar/checkpoints/mmdetection_work_dirs/virat_cascade_personobject/cascade_personobject_val_allframes_32step16

#!/usr/bin/env bash

# bash tools/dist_train.sh configs/0cd_ce_baseline_me/changeclip_whucd.py 4 --work-dir work_dirs_baseline_me/changeclip_whucd
# bash tools/test.sh WHUB configs/0seg/baseline_whub.py 4 work_dirs/baseline_whub

# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub.py 4 work_dirs/mmseg_text_whub

# bash tools/dist_train.sh configs/0seg/mmseg_text_whub.py 2 --work-dir work_dirs/mmseg_text_whub_v2
# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub.py 2 work_dirs/mmseg_text_whub_v2


# bash tools/dist_train.sh configs/0seg/mmseg_text_whub_vit.py 2 --work-dir work_dirs/mmseg_text_whub_vit
# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub_vit.py 2 work_dirs/mmseg_text_whub_vit

# bash tools/dist_train.sh configs/0seg/mmseg_text_whub_v2.py 2 --work-dir work_dirs/mmseg_text_whub_v2
# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub_v2.py 2 work_dirs/mmseg_text_whub_v2

bash tools/train.py configs/0seg_ab/baseline.py 2 --work-dir work_dirs_ab/baseline
bash tools/test.sh WHUB configs/0seg_ab/baseline.py 2 work_dirs_ab/baseline
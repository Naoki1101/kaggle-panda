cd ../src

# =============================================================================
# Classification
# =============================================================================
# python train.py -m 'clf_efficientnet_b3' -c '...'

# python train.py -m 'clf_resnest50' -c '...'

# python train.py -m 'clf_resnet18' -c 'img_size=1280'
# python train.py -m 'clf_resnet34' -c 'img_size=1536'
# python train.py -m 'clf_resnet50' -c '...'

# python train.py -m 'clf_se_resnext50_32x4d' -c 'SmoothCrossEntropyLoss'


# =============================================================================
# Regression
# =============================================================================
# python train.py -m 'reg_efficientnet_b0' -c 'test'

# python train.py -m 'reg_ghostnet' -c 'test'

# python train.py -m 'reg_resnest50' -c '...'

# python train.py -m 'reg_resnet18' -c 'test tta'
# python train.py -m 'reg_resnet34' -c 'use top16 tiles'
# python train.py -m 'reg_resnet34_2' -c 'noise_idx'
# python train.py -m 'reg_resnet34_3' -c 'duplicate_img_idx, noise_idx'
# python train.py -m 'reg_resnet34_4' -c 'duplicate_img_idx, noise_idx, large_diff_idx'
# python train.py -m 'reg_resnet34_5' -c '128x144 duplicate_img_idx, noise_idx, large_diff_idx'
# python train.py -m 'reg_resnet34_6' -c '128x144, large_diff_idx'
# python train.py -m 'reg_resnet34_7' -c '384x16, duplicate_img_idx, noise_idx'
# python train.py -m 'reg_resnet34_8' -c '128x144, large'
# python train.py -m 'reg_resnet34_9' -c '128x144, duplicate & noise'
# python train.py -m 'reg_resnet34_10' -c '128x144, large & duplicate & noise'
# python train.py -m 'reg_resnet34_11' -c '256x36, large'
# python train.py -m 'reg_resnet34_12' -c '256x36, duplicate & noise'
# python train.py -m 'reg_resnet34_13' -c '256x36, large & duplicate & noise'
# python train.py -m 'reg_resnet34_14' -c '384x16, large'
# python train.py -m 'reg_resnet34_15' -c '384x16, duplicate & noise'
# python train.py -m 'reg_resnet34_16' -c '384x16, large & duplicate & noise'

python train.py -m 'reg_resnet34_17' -c 'seed=2021, large_diff_idx'
python train.py -m 'reg_resnet34_18' -c 'seed=2021, noise_idx'
python train.py -m 'reg_resnet34_19' -c 'seed=2021, duplicate_img_idx, noise_idx'
python train.py -m 'reg_resnet34_20' -c 'seed=2021, duplicate_img_idx, noise_idx, large_diff_idx'
python train.py -m 'reg_resnet34_21' -c 'seed=2021, 128x144 duplicate_img_idx, noise_idx, large_diff_idx'

# python train.py -m 'reg_resnet50' -c 'img_size=c'

# python train.py -m 'reg_se_resnext50_32x4d' -c 'use top9(simple) tiles'


# =============================================================================
# Ordinal Regression
# =============================================================================
# python train_ordinal_reg.py -m 'ordinal_reg_resnet34' -c '...'


cd ../
git add -A
git commit -m '...'
git push origin master

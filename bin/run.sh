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
# python train.py -m 'reg_resnet34' -c 'large_diff_idx'
# python train.py -m 'reg_resnet34_2' -c 'noise_idx'
# python train.py -m 'reg_resnet34_3' -c 'duplicate_img_idx, noise_idx'
# python train.py -m 'reg_resnet34_4' -c 'duplicate_img_idx, noise_idx, large_diff_idx'
# python train.py -m 'reg_resnet34_5' -c '128x144 duplicate_img_idx, noise_idx, large_diff_idx'
# python train.py -m 'reg_resnet34_6' -c '128x144, large_diff_idx'
python train.py -m 'reg_resnet34_7' -c '384x16, duplicate_img_idx, noise_idx'

# python train.py -m 'reg_resnet50' -c 'img_size=c'

# python train.py -m 'reg_se_resnext50_32x4d' -c 'img_size=768'


# =============================================================================
# Ordinal Regression
# =============================================================================
# python train_ordinal_reg.py -m 'ordinal_reg_resnet34' -c '...'


cd ../
git add -A
git commit -m '...'
git push origin master

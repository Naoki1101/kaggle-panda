cd ../src

# =============================================================================
# Classification
# =============================================================================
# python train.py -m 'clf_efficientnet_b3' -c '...'

# python train.py -m 'clf_resnest50' -c '...'

# python train.py -m 'clf_resnet18' -c 'img_size=1280'
# python train.py -m 'clf_resnet50' -c '...'

# python train.py -m 'clf_se_resnext50_32x4d' -c 'SmoothCrossEntropyLoss'


# =============================================================================
# Regression
# =============================================================================
# python train.py -m 'reg_efficientnet_b0' -c 'img_size=1536'

# python train.py -m 'reg_ghostnet' -c 'test'

# python train.py -m 'reg_resnest50' -c '...'

# python train.py -m 'reg_resnet18' -c 'drop duplicate(new)'
python train.py -m 'reg_resnet34' -c 'MSELoss'
# python train.py -m 'reg_resnet50' -c 'img_size=c'

# python train.py -m 'reg_se_resnext50_32x4d' -c 'img_size=768'


cd ../
git add -A
git commit -m '...'
git push origin master

cd ../src
# python train.py -m 'clf_efficientnet_b3' -c '...'
# python train.py -m 'clf_resnest50' -c '...'
# python train.py -m 'clf_resnet18' -c 'test imgae_mask'
# python train.py -m 'clf_resnet50' -c '...'
python train.py -m 'clf_se_resnext50_32x4d' -c 'test imgae_mask'


cd ../
git add -A
git commit -m '...'
git push origin master

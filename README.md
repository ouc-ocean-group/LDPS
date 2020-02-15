# Learning Deep Relations to Promote Saliency Detection

This is the code repo for "Learning Deep Relations to Promote Saliency Detection", accepted by AAAI2020.

## Train
```bash
python3 train_mi.py --backbone mobilenetv2 --batch-size 1 --lr 0.0005 --prior picanet --iter-num 20000 --t-fg 0.9 --t-afg 0.8 --t-abg 0.3 --test-prior srm --test-dataset ecssd 
```

## Test
```bash
python3 test_mi.py --sync-bn False --distributed False --test-dataset DATASET --test-prior PRIOR --ckp ./ckp/xxxx.pth --test True --save-result True
```

## Citing This Paper
If you find this work useful in your research, please consider citing:

```
@inproceedings{chen2020r,
  title={Learning Deep Relations to Promote Saliency Detection},
  author={Chen, Changrui and Sun, Xin and Hua, Yang and Dong, Junyu and Xv, Hongwei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
# README
Implementation of a classifier of real an fake images, based on [SimSiam paper](https://arxiv.org/abs/2011.10566). Please refer to the [original repo](https://github.com/facebookresearch/simsiam/tree/main) if you have to use the original implementation of the SimSiam model.

## Warning
This version of simsiam training has been adapted to work with the [TrueFace dataset](https://mmlab.disi.unitn.it/resources/published-datasets#h.4bwcjdyr0h5i), respecting its folder structure. Please refer to the original repository if you want to use simsiam with more general purpose datasets.

## Installation
All python dependencies are listed inside `requirements.txt`. To install them all, use `python -m pip install -r requirements.txt`

## Running the pretrain
A complete pretrain of the network can be run with :
```
python main_simsiam.py <path-to-dataset> --train-prepost -a=resnet50 --world-size=1 --rank=0 --gpu=0 --epochs=120 --real-amount=7000 --fake-amount=7000 --batch-size=64 --workers=8 --lr=0.1 --save-checkpoints --save-dir='../pretrained/'
```
Use `--train-prepost` to train on both presocial and postsocial dataset, or specify the desired subset folder in the path given with the `--data` argument to train only on the presocial or postsocial dataset.
## Running the finetuning
A finetuning of the network can be run with:
```
python main_lincls.py <path-to-dataset> -a=resnet50 --world-size=1 --rank=0 --gpu=0 --workers=8 --epochs=100 --real-amount=7000 --fake-amount=7000 --batch-size=64 --pretrained=<path-to-checkpoint> --lr=0.1 --checkpoint-name=finetuned.pth.tar
```
## Testing the weights
This command will test the network against all images in the 'Test' subset of TrueFace.
```
python test_classifier.py <path-to-test-set> -a=resnet50 --world-size=1 --rank=0 --gpu=0 --real-amount=1500 --fake-amount=1500 --batch-size=32 --workers=8 --finetuned=finetuned.pth.tar --save-precise-report --conf-matrix                              
```

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
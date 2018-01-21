# DNN-Pruning

Pruing is one of the hot topics on reduction of computing resources requiered by ML-based applications. The literature is now plentyful on different criteria that can be used to select what weights to prune. In their papaer [Pruning Filter for Efficient ConvNets](https://arxiv.org/abs/1608.08710), Li Hao and his co-authors intoduce a simple approach for pruning CNNs without needing to evaluate forward path on a dataset.  A nice and read-friendly blog on the subject of pruning can be found in this [blog](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

## Dependencies

The base of this code was written and shared by [He Yihui](https://github.com/yihui-he), and runs on his custom branch of CAFFE (python 3 implementation).
After cloning the repo, build the CAFFE binaries by following the instructions in the branch [repo](https://github.com/yihui-he/caffe-pro).
Train or donwload a caffemodel and place it in the temp directory. 
Make sure to modify cfgs.py with the layer names and pruning degree (i.e. number of channels that will be retained after the pruning process) you desire for each layer.

## Usage


Run: 
```
python3 filter_pruning_demo.py -model ./temp/model.prototxt -weights ./temp/model.caffemodel
```
The scripts offers the option for pruning the model OR plotting the sensitivity curves, which are usefull to determine what layers can be pruned more agressively without much accuracy degradation.

Follow the command prompt instruction to select the operation mode or the pruning degree when asked.

## Example

When adjusting the code to plot the sensitivity curves pruning 1 channel at a time, the following graphs like the following can be obtained:

- VGG16 model
- CIFAR-10 dateset

![vgg16-cifar10-detailed](https://user-images.githubusercontent.com/24645932/35195504-fc8bc1fc-ff07-11e7-8982-afe6fe39c5fa.png)

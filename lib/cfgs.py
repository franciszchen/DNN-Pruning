from easydict import EasyDict as edict

##############################################################################
# Load this dictionary with the layer names and/or the desired number of
# channels per layer to keep.
c=edict()# TODO: easy dict does not have order. Try using Ordered Dict to print dc cfgs in order
c.noLastConv = False # Set this flag to True to avoid last layer pruning.
                     #In some models like vgg-cifar, the last-layer pruning yeilds an error when saving to protobuffer

# dcdic holds the number of retained channels
# For example, for VGG net we can use the pruning degree suggested in (Li et.al., 2017)
c.dcdic = {'conv1_1': 32,'conv1_2': 64,'conv2_1': 128,'conv2_2': 128,'conv3_1': 256,'conv3_2': 256,'conv3_3': 256,'conv4_1': 256,'conv4_2': 256,'conv4_3': 256,'conv5_1': 256,'conv5_2': 256,'conv5_3': 256}
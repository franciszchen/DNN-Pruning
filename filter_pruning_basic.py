from __future__ import print_function
from easydict import EasyDict as edict
from lib.cfgs import c as dcfgs
import lib.cfgs as cfgs
import os
os.environ['JOBLIB_TEMP_FOLDER']=dcfgs.shm
import argparse
os.environ['GLOG_minloglevel'] = '3'
import os.path as osp
import pickle
import sys
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from lib.decompose import *
from lib.net import Net, load_layer, caffe_test
from lib.utils import *
from lib.worker import Worker
import google.protobuf.text_format # added to fix missing protobuf properties -by Mario
sys.path.insert(0, osp.dirname(__file__)+'/lib')
#import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
from matplotlib.ticker import FormatStrFormatter


import caffe


# ----- Main Functions ----

def loadModel (pt, model, WPQ = None, selection = None):
    """
    This function simply instantiates the net object
    and returns the net objets.
    """

    if DEBUG_Mario:
        print("\n--- load model ---")
        print("in pt: ", pt)
        print("in model: ", model)

    net = Net(pt, model=model, phase=caffe.TEST)
    net.WPQ = dict()
    net.selection = dict()

    if WPQ is not None:
        net.WPQ = WPQ
    if selection is not None:
        net.selection = selection

    if DEBUG_Mario:
        print("\nout net : 'prune_me' ")

    return net

def finetune():
    """
    #TODO: It might be bood to prune several times in small steps
    with finetune in between pruning steps.
    Implementing thisis complicated because the prototxt that we load at
    each step should have the layer names changed or else caffe will fail to copy
    from the .caffemodel
    Consider adding a postfix when adding parameters to WPQ at each pruning step.
    """

    pass

def setProto(net, prefix):
    """
    This is an auxiliary function that takes the pruned model, sets up the new .prototxt and extracts the WPQ dict
    (Actions taken by step4 used to be part of the net.R3 methods)
     """
    if DEBUG_Mario:
        print("\n--- set Proto ---")
        print("in net's pt  ", net.pt_dir)

    new_pt = net.save_pt(prefix=prefix)
    WPQ = net.WPQ

    if DEBUG_Mario:
        print("\nout pt (new_pt) : ", new_pt)
        print("out WPQ  : ", WPQ.keys())


    return new_pt, WPQ

def saveModel(new_pt, model, WPQ):
    """
    This functions writes to the protobuf to
    generate the new .caffemodel
    """

    if DEBUG_Mario:
        print("\n--- saveModel ---")
        print("in pt (new_pt): ", new_pt)
        print("in model: ", model)
        print("in WPQ: ", WPQ.keys())

    net = Net(new_pt, model=model)
    net.WPQ = WPQ
    net.finalmodel(save=False) # model
    new_pt, new_model = net.save(prefix='pruned')

    if DEBUG_Mario:
        print("\nout pt (new_pt): ", new_pt)
        print("out model (new_model): ", new_model)

    print('\nFinal model ready. For testing you can use:' )
    print('\t$CAFFE_ROOT/build/tools/caffe test -model',new_pt, '-weights',new_model)
    return new_pt, new_model

# ----- Pruning Functions -----

def pruneLayer(net, conv, convnext, bn , affine, dc):

    conv_P = underline(conv,'P')
    if DEBUG_Mario:
        print("-"*30); print("\t Filter Pruning "); print("-"*30)
        print("operating: conv = %s , convnext = %s, bn = %s, dc = %s" % (conv,convnext,bn,dc))


    if conv in net.selection:
        if DEBUG_Mario:
            print(" @  Pruning filter chanels (W2 from last iter) @")
            print(" Reshape WPQ[(conv,0)] from: ",net.param_data(conv).shape )
        net.WPQ[(conv_P,0)] =  net.param_data(conv)[:,net.selection[conv],:,:]
        net.WPQ[(conv_P,1)] =  net.param_b_data(conv)
        if DEBUG_Mario:
            print("to:                          ",net.WPQ[(conv_P,0)].shape )
            print("Biases shape: ",net.WPQ[(conv_P,1)].shape)
    else:
        net.WPQ[(conv_P,0)] =  net.param_data(conv)
        net.WPQ[(conv_P,1)] =  net.param_b_data(conv)
        if DEBUG_Mario:
            print(" @ First iter(W2)  @")
            print("shape of WPQ[(conv,0)] : ",net.WPQ[(conv_P,0)].shape )
            print("shape of WPQ[(conv,1)] : ",net.WPQ[(conv_P,1)].shape )


    weights = net.param_data(conv)
    idxs, W2, _ = net.pruning_kernel(None,weights, dc, convnext, None)

    if DEBUG_Mario:
            print("--- pruning_kernel() ret ---")
            print("idxs" , idxs.shape)
            print("W2" , W2.shape)

    # W2
    net.selection[convnext] = idxs
    net.param_data(convnext)[:, ~idxs, ...] = 0
    net.param_data(convnext)[:, idxs, ...] = W2.copy()


    if DEBUG_Mario:
        print("@ Set W2 @")
        print("weights convnext that are set to zero: ", net.param_data(convnext)[:, ~idxs, ...].shape)
        print("weights convnext that are retained   : ", net.param_data(convnext)[:, idxs, ...].shape)
        print("weights covnext in model             : ", net.param_data(convnext).shape)


    # W1
    net.WPQ[(conv_P,0)] = net.WPQ[(conv_P,0)][idxs]
    net.WPQ[(conv_P,1)] = net.WPQ[(conv_P,1)][idxs]
    net.set_conv(conv, num_output=sum(idxs),new_name=conv_P)
    if DEBUG_Mario:
        print("@ Prune filters W1 @")
        print("net.WPQ[(conv,0)] ",net.WPQ[(conv_P,0)].shape)
        print("net.WPQ[(conv,1)] ",net.WPQ[(conv_P,1)].shape)

   # W1's BN
    if bn is not None:
        bn_P = underline(bn,'P')
        net.WPQ[bn_P] = net.param_data(bn)[idxs]
        net.set_bn(bn, new_name=bn_P)

        affine_P = underline(affine,'P')
        net.WPQ[affine_P] = net.param_data(affine)[idxs]
        net.set_bn(affine, new_name=affine_P)

        if DEBUG_Mario:
            print("@ Prune BN layer between %s and %s @" %(conv,convnext))
            print("net.WPQ[bn_P] ", net.WPQ[bn_P].shape)
            print("net.WPQ[affine_P] ", net.WPQ[affine_P].shape)

    return net

def pruneLastLayer(net,conv, bn, affine, dc):

    if DEBUG_Mario:
        print("-"*30); print("\t Filter Pruning  "); print("-"*30)
        print("Last run, operating: conv = %s , bn = %s, dc = %s" % (conv,bn,dc))
    conv_P = underline(conv,'P')

    if DEBUG_Mario:
            print(" @  Pruning filter chanels (W2 from last iter) @")
            print(" Reshape WPQ[(conv,0)] from: ",net.param_data(conv).shape )
    net.WPQ[(conv_P,0)] =  net.param_data(conv)[:,net.selection[conv],:,:]
    net.WPQ[(conv_P,1)] =  net.param_b_data(conv)
    if DEBUG_Mario:
        print("to:                          ",net.WPQ[(conv_P,0)].shape )
        print("Biases shape: ",net.WPQ[(conv_P,1)].shape)


    if not noLastConv:
        weights = net.param_data(conv)

        idxs = np.argsort(-np.abs(weights).sum((1,2,3)))
        idxs = np.sort(idxs[:dc])
        newidxs = np.zeros(len(weights)).astype(bool)
        newidxs[idxs] = True

        # W1
        net.WPQ[(conv_P,0)] = net.WPQ[(conv_P,0)][newidxs]
        net.WPQ[(conv_P,1)] = net.WPQ[(conv_P,1)][newidxs]
        net.set_conv(conv, num_output=sum(newidxs),new_name=conv_P)
        if DEBUG_Mario:
            print("@ Prune filters W1 @")
            print("net.WPQ[(conv,0)] ",net.WPQ[(conv_P,0)].shape)
            print("net.WPQ[(conv,1)] ",net.WPQ[(conv_P,1)].shape)

        # W1's BN
        if bn is not None:
            bn_P = underline(bn,'P')
            net.WPQ[bn_P] = net.param_data(bn)[newidxs]
            net.set_bn(bn, new_name=bn_P)

            affine_P = underline(affine,'P')
            net.WPQ[affine_P] = net.param_data(affine)[newidxs]
            net.set_bn(affine, new_name=affine_P)

            if DEBUG_Mario:
                print("@ Prune BN layer after  %s @" % conv)
                print("net.WPQ[bn_P] ", net.WPQ[bn_P].shape)
                print("net.WPQ[affine_P] ", net.WPQ[affine_P].shape)

        if DEBUG_Mario:
            print("-"*30); print("\t Prune first FC layer "); print("-"*30)

        FCs = net.innerproduct
        fc = FCs[0] # get the nae of the first FC layer
        fc_P = underline(fc,'P')
        fc_bottom_shape = prune_me.blobs_shape(prune_me.layer_bottom(fc))
        if len(fc_bottom_shape) == 4:
            unit_serial_len = fc_bottom_shape[2] * fc_bottom_shape[3]
        elif len(fc_bottom_shape) == 3:
            unit_serial_len = fc_bottom_shape[2]

        if unit_serial_len != 1:
            fc_newidxs = np.array([], dtype=np.bool)
            for i in range(len(newidxs)):
                fc_newidxs = np.append(fc_newidxs, np.array([newidxs[i]] * unit_serial_len))
        else:
            print('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('@ unit_serial_len is 1. This code will probably not be able to prune the last conv layer.     @')
            print('@ Use the flag <noLastConv> in the cfgs file to prevent last layer from being pruned. Sorry TT@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            fc_newidxs = newidxs

        net.WPQ[fc_P] = net.param_data(fc)[:,fc_newidxs]
        net.set_fc(fc, new_name=fc_P)
        if DEBUG_Mario:
                print("@ Prune inputs of %s layer  @" % fc)
                print("net.WPQ[fc_P] ", net.WPQ[fc_P].shape)
                print("\nThanks to Minsu for solving the fc issue!")
    else:
        net.set_conv(conv, new_name=conv_P)

    return net

# ----- Auxiliary Functions -----

def loadConfigs(pt, model):

    if DEBUG_Mario:
         print("\n--- load configs ---")
         print("in pt:", pt)
         print("in model:", model)

    dcdic = dcfgs.dcdic
    print("\nCurrent pruning configuration (loaded from lib/cfgs.c.dcdic)")
    for key,value in dcdic.items():
        print(key,value)

    changeConfigs= checkOperation(question= '\nModify the prunig configuration?',default='no')

    if changeConfigs:
        net = Net(pt, model=model, phase=caffe.TEST)
        convs = net.convs

        for conv in convs:
            print('input number remaining filters of %s after pruning [original number of filters %d ]' % (conv, net.conv_param_num_output(conv)))
            dc = input("[leave blank to skip]:  ")
            if dc =='':
                dcdic[conv] = dcdic[conv]
            else:
                dcdic[conv] = int(dc)
    return dcdic

def checkOperation(question = 'Pass a question string to checkOperation!', default='no'):

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Valid answers: 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def registerBNs(net):
    BNs = net.bns#list with name of the BatchNorm layers
    Affines = net.affines
    FCs = net.innerproduct
    assert len(BNs) == len(Affines)

    if not BNs:
        print("no BN layers found in %s" % net.pt_dir)
        net.noBNs = True #there is no BN layers
        BNs_dic = None
    else:
        print("BN layers found in %s" % net.pt_dir)
        net.noBNs = False
        BNs_dic = {}
        for BN, Affine in zip(BNs,Affines):
            BN_bottom= net.bottom_names[BN][0] # of name of the bottoms of each BN layers
            if BN_bottom not in FCs:
                BNs_dic[BN_bottom] = [BN,Affine] # dictionary that holdes bottom and name of each BN and scale layer

        if DEBUG_Mario:
            for key,value in BNs_dic.items():
                print(key,value)


    return net, BNs_dic

def checkIfBN(noBNs, conv, bns_dic):
    if not noBNs: #if there is BN layers in the Net
        try:
            bn = bns_dic[conv][0]
            affine = bns_dic[conv][1]
            return bn, affine

        except KeyError:
            print("%s layer does not have BN" % conv)
            return None, None
    else:
        return None, None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='prototxt directory', default=None, type=str)
    parser.add_argument('-weights', help='caffemodel directory', default=None, type=str)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

# ----- Utility Functions -----

def printWPQdic(WPQ):

    print("Dictionary stored data: ")
    for key, value in WPQ.items():
        print(key, value.shape)

def printweights(net):
    print("weights\t\tbias")
    for conv in net.convs:
        print(net.param_shape(conv), net.param_b_shape(conv))

def saver(filename,data):
    #save plot data
    ROOT_DIR = 'temp/'
    FILENAME = filename

    SAVE_DIR = osp.join(ROOT_DIR, FILENAME + ".pickle")

    os.makedirs(osp.dirname(SAVE_DIR), exist_ok=True)

    with open(SAVE_DIR, 'wb') as f:
        print("\nSaving plot data to: ", SAVE_DIR)
        pickle.dump(data, f)

    print("...done")

def loader(filename):
    #load plot data
    ROOT_DIR = 'temp/'
    FILENAME = filename

    LOAD_DIR = osp.join(ROOT_DIR, FILENAME + ".pickle")

    with open(LOAD_DIR, 'rb') as f:
        print("\nLoading plot data from: ", LOAD_DIR)
        data = pickle.load(f)
        print("...done")
    return data

# ----- Sensitivity Functions -----

def pseudoPruneLayer(net, conv, convnext, bn , affine, dc):

    if DEBUG_Mario:
        print("-"*30); print("\t Filter Pseudo Pruning "); print("-"*30)
        print("operating: conv = %s , convnext = %s, bn = %s, dc = %s" % (conv,convnext,bn,dc))


    weights = net.param_data(conv)
    idxs, W2, _ = net.pruning_kernel(None,weights, dc, convnext, None)

    if DEBUG_Mario:
            print("--- pruning_kernel() ret ---")
            print("idxs" , idxs.shape)
            print("W2" , W2.shape)

    # W2
    net.param_data(convnext)[:, ~idxs, ...] = 0
    net.param_data(convnext)[:, idxs, ...] = W2.copy()


    if DEBUG_Mario:
        print("@ Set W2 @")
        print("weights convnext that are set to zero: ", net.param_data(convnext)[:, ~idxs, ...].shape)
        print("weights convnext that are retained   : ", net.param_data(convnext)[:, idxs, ...].shape)
        print("weights covnext in model             : ", net.param_data(convnext).shape)


    # W1
    net.param_data(conv)[~idxs] = 0
    net.param_b_data(conv)[~idxs] = 0

    if DEBUG_Mario:
        print("@ Prune filters W1 @")
        print("weights conv that are set to zero: ", net.param_data(conv)[~idxs].shape)
        print("weights conv that are retained   : ", net.param_data(conv)[idxs].shape)
        print("weights conv in model            : ",net.param_data(conv).shape)

   # W1's BN
    if bn is not None:

        net.param_data(bn)[~idxs] = 0
        net.param_data(affine)[~idxs] = 0

        if DEBUG_Mario:
            print("@ Prune BN layer between %s and %s @" %(conv,convnext))
            print("weigths of the BN layer that are retained ", net.param_data(bn)[idxs].shape)
            print("wieghts of the Scaling layers that are retained ", net.param_data(affine)[idxs].shape)

    return net

def pseudoPruneLastLayer(net,conv, bn, affine, dc):

    if DEBUG_Mario:
        print("-"*30); print("\t Filter Pseudo Pruning "); print("-"*30)
        print("operating: conv = %s , bn = %s, dc = %s" % (conv,bn,dc))

    if DEBUG_Mario: print("Last run, prune weights of %s " % conv)


    weights = net.param_data(conv)

    idxs = np.argsort(-np.abs(weights).sum((1,2,3)))
    idxs = np.sort(idxs[:dc])
    newidxs = np.zeros(len(weights)).astype(bool)
    newidxs[idxs] = True

    # W1
    net.param_data(conv)[~newidxs] = 0
    net.param_b_data(conv)[~newidxs] = 0

    if DEBUG_Mario:
        print("@ Prune filters W1 @")
        print("weights conv that are set to zero: ", net.param_data(conv)[~newidxs].shape)
        print("weights conv that are retained   : ", net.param_data(conv)[newidxs].shape)
        print("weights conv in model            : ",net.param_data(conv).shape)

    # W1's BN
    if bn is not None:

        net.param_data(bn)[~newidxs] = 0
        net.param_data(affine)[~newidxs] = 0

        if DEBUG_Mario:
            print("@ Prune BN layer after %s " % conv)
            print("weigths of the BN layer that are retained ", net.param_data(bn)[newidxs].shape)
            print("wieghts of the Scaling layers that are retained ", net.param_data(affine)[newidxs].shape)


    if DEBUG_Mario:
        print("-"*30); print("\t Pseudo Prune first FC layer "); print("-"*30)

    FCs = net.innerproduct
    fc = FCs[0] # get the name of the first FC layer

    #get first FC layers bottom shape, get one serialized filter length for one neuron
    fc_bottom_shape = net.blobs_shape(net.layer_bottom(fc))
    if len(fc_bottom_shape) == 4:
        unit_serial_len = fc_bottom_shape[2] * fc_bottom_shape[3]
    elif len(fc_bottom_shape) == 3:
        unit_serial_len = fc_bottom_shape[2]

    # reshape the index to the serialized weight filter
    if unit_serial_len != 1:
        fc_newidxs = np.array([], dtype=np.bool)
        for i in range(len(newidxs)):
            fc_newidxs = np.append(fc_newidxs, np.array([newidxs[i]] * unit_serial_len))
    else:
        fc_newidxs = newidxs

    net.param_data(fc)[:,~fc_newidxs] = 0

    if DEBUG_Mario:
            print("@ Prune inputs of %s layer  @" % fc)
            print("weights of fc that are retained ", net.param_data(fc)[:,fc_newidxs].shape)

    return net

def accuracy(net,n_batches):
    acc1 = []
    acc5 = []
    for i in range(n_batches):
        res = net.net.forward()
        acc1.append(float(res['accuracy@1']))
        acc5.append(float(res['accuracy@5']))

    macc1 = np.mean(acc1)
    macc5 = np.mean(acc5)
    #if DEBUG_Mario: print("top1 acc: %3.3f , top5 acc: %3.3f" % (macc1, macc5))
    return macc1, macc5

def plotter(data, model_set,save=True, legendloc='lower left'):

    for conv in data.keys():
        dc_init = data[conv][0][0]
        x = np.array(data[conv][0])
        x = x/dc_init # normalize (percentage or remaining filters)
        x = 100*(1-x) # percentage of pruned filters

        y1 = 100*np.array(data[conv][1])
        #y5 = np.array(p_dic[conv][2])

        real1 = plt.plot(x,y1,label=conv)

    m_s = model_set.split('-')
    MODEL = m_s[0]
    SET = m_s[1]
    TITLE = 'Pruning Sensitivity - '+ '('+ MODEL +',' + SET+ ')'
    plt.legend(loc=legendloc)
    plt.xlabel('Pruned Filters (%)')
    plt.title(TITLE)
    plt.ylabel('Top1 Val. Accuracy (%)')

    plt.grid()
    if save:
        dir_path = osp.dirname(osp.realpath(__file__))
        print("\nSaving %s.png to %s directory..." % (model_set,dir_path))
        plt.savefig(model_set, dpi=600, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=True, bbox_inches='tight', pad_inches=0.1,
                frameon=None)
        print("...done")
    #plt.show()

def checkSensitivity(pt,model,graph_title='<Model>-<Dataset>'):
    """
    This funtion prunes the model layer by layer, from 100% to
    100/p_Nx % of its filter. It then plots the sensibility curves and
    saves the plot data to a pickle file for later analysis.

    NOTE: The pruning performed here does not alter the model shape, simply
    makes filters zero, enough for accuracy evaluation.

    """

    p_dic={}#ploting dictionary
    p_Nx = 20 #number of points to plot (step for decreasing % of prune weights; 5% in this case)
    n_batches = 100 #number of test batches


    net = Net(pt, model=model, noTF=1)# ground truth net
    convs = net.convs
    net, BNs_dic = registerBNs(net)

    noBNs = net.noBNs

    #initial accuracy test
    print('@ Grand-truth model @')
    acc1_init , acc5_init = accuracy(net, n_batches)

    for conv, convnext in zip(convs[0:], convs[1:]):

        net = Net(pt, model=model, noTF=1) #reset the model
        bn, affine = checkIfBN(noBNs, conv, BNs_dic)

        dc_init = net.conv_param_num_output(conv)
        dc = dc_init
        p_x = [dc_init] #plot points of x_axis
        p_acc1 = [acc1_init]
        p_acc5 = [acc5_init]


        for i in range(p_Nx-1):#the first point was already added (initial conditions), reduce range by 1
            dc = round(dc - dc_init/p_Nx)
            net = pseudoPruneLayer(net, conv, convnext, bn, affine, dc)

            acc1, acc5 = accuracy(net, n_batches)

            p_x.append(dc)
            p_acc1.append(acc1)
            p_acc5.append(acc5)

        if DEBUG_Mario: print("Layer %s done.", conv)

        p_dic[conv] = [p_x,p_acc1,p_acc5]# load the dictionary with the ploting data


        if convnext == convs[-1]:# If conv is the last of the list prune convnext before leaving the loop
                              #(convnext will be the last pruned layer)

            net = Net(pt, model=model, noTF=1) #reset the model
            bn, affine = checkIfBN(noBNs, convnext, BNs_dic)

            dc_init = net.conv_param_num_output(convnext)
            dc = dc_init
            p_x = [dc_init] #plot points of x_axis
            p_acc1 = [acc1_init]
            p_acc5 = [acc5_init]

            for i in range(p_Nx-1):#the first point was already added (initial conditions), reduce range by 1
                dc = round(dc - dc_init/p_Nx)
                net = pseudoPruneLastLayer(net, convnext,  bn, affine, dc)

                acc1, acc5 = accuracy(net, n_batches)

                p_x.append(dc)
                p_acc1.append(acc1)
                p_acc5.append(acc5)

            if DEBUG_Mario: print("Last layer %s done.", convnext)

            p_dic[convnext] = [p_x,p_acc1,p_acc5]# load the dictionary with the ploting data
     ### End of the for-loop####

    saver(graph_title, p_dic)

    return p_dic

def sensitivityScore(graph_title):
    """
    Creates a sensibility scorefor each layer based on the
    sensibility curves
    """

    def calculateScore1(y):
        """
        convolution
        NOTE: Scores calculated with this criteria seem to be unreliable
        --- Tested on: bn_alexnet, imagenet
        """
        yl = int(len(y))
        ylh = int(yl/2)


        score = np.sum( (y[-ylh:]-y[-yl:-ylh]) / yl )
        score = np.absolute(score)
        return score

    def calculateScore2(y):
        """integral"""

        score = np.trapz(y)

        return score


    plot_data = loader(graph_title)

    scores = {}
    for key, value in plot_data.items():

        data = np.array(value[1])
        #scores[key]=calculateScore1(data)
        scores[key]=calculateScore2(data)

    return scores

if __name__ == '__main__':

    DEBUG_Mario = 1
    args = parseArgs()
    noLastConv = dcfgs.noLastConv

    pt = args.model
    model = args.weights

    prune_mode = checkOperation(question='Prune model? [Type YES to continue or NO to plot sensitivity] ',default='yes')

    if prune_mode:# Prune Model
        print("\nOperation mode: Pruning")

        dcdic = loadConfigs(pt, model)

        prune_me = loadModel(pt,model)
        convs = prune_me.convs

        test_n_batches = 50
        gt_acc1 , gt_acc5 = accuracy(prune_me, test_n_batches )

        print("\nPairs of layers that will be pruned:")
        for conv, convnext in zip(convs[0:], convs[1:]):
            print(conv,convnext)

        proceed = checkOperation("\ncontinue? ", default='no')
        if not proceed: sys.exit()

        prune_me, BNs_dic = registerBNs(prune_me)

        for conv, convnext in zip(convs[0:], convs[1:]):

            bn, affine = checkIfBN(prune_me.noBNs, conv, BNs_dic)
            dc = dcdic[conv]


            #prune a hidden layer
            prune_me = pruneLayer(prune_me, conv, convnext, bn, affine, dc)

            ######################################################
            #finetune after each layer pruning?
            #TODO: Quck fine-tuning after layer pruning
            #NOTE: In the paper they prune the hold model before
            #finetuning
            ######################################################


            #prune last layer (the standard pruneLayer() function omits this layer)
            if convnext == convs[-1]:

                bn, affine = checkIfBN(prune_me.noBNs, convnext, BNs_dic)
                dc = dcdic[convnext]

                prune_me = pruneLastLayer(prune_me, convnext, bn, affine, dc)

        new_pt, WPQ = setProto(prune_me,prefix='renamed')

        #save the final .caffemodel
        new_pt, new_model = saveModel( new_pt, model, WPQ )

        #Load the pruned model for testing
        print("\n runing accuracy test of the pruned model (%d test batches)" % test_n_batches)
        prune_me = loadModel(new_pt, new_model)
        acc1 , acc5 = accuracy(prune_me, test_n_batches)
        print("Ground-truth accuracy:                top1 %3.3f , top5 acc %3.3f" % (gt_acc1, gt_acc5))
        print("Pruned model accuracy: (no fine-tune): top1 %3.3f , top5 acc %3.3f" % (acc1, acc5))

    else: # Plot Sensitibity
        print("Operation mode: Sensitivity Analysis")
        model_set = input("Type in the graph title in the format <Model name-dataset name>  :")
        plot_data = checkSensitivity(pt,model,model_set)
        plotter(plot_data, model_set, save=True, legendloc='lower left')

        scores = sensitivityScore(model_set)
        print('-'*60)
        print("Sensitibity scores")
        for key,value in scores.items():
            print(key,value)

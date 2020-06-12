from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from sklearn.cluster import KMeans
import statistics


def prune(model):
    state_dict = model.state_dict()
    threshold = .05
    print("\nThreshold: ", threshold)
    nonzero_sum = 0
    total_sum = 0
    for key, value in state_dict.items():
        # if ("weight" in key or "bias" in key):  
        if ("weight" in key):
            copy = value.flatten()
            copy, indices = copy.sort()
            print(key)
            print("Min:", copy.data[0].item(), " Max: ", copy.data[-1].item())
            mask = abs(value) > threshold 
            new_value =  mask*value
            state_dict[key] = new_value
            nonzero_sum += mask.sum().item()
            total_sum += value.numel()
    model.load_state_dict(state_dict)
    print("Percent Remaining: ", nonzero_sum/total_sum)
    return

def share_weights_pruned(model):
    state_dict = model.state_dict()
    weight_key = False
    c = 0
    for key, value in state_dict.items():
        
        if "orig" in key:
            weight_key = key
            
         
        if "mask" in key:
            mask = value.cpu().numpy()
            mask_flatten = mask.flatten()
            

            
            weights = state_dict[weight_key].cpu().numpy()
            weights_flatten = weights.flatten()
            assert(mask.shape == weights.shape)
            
            ## obtain a list of all unmasked weights
            unmasked_weights = []
            for i, mask_val in enumerate(mask_flatten):
                if mask_val:
                    unmasked_weights.append(weights_flatten[i])
            unmasked_weights = np.asarray(unmasked_weights)
            
            ## get the min and max value of the umasked weights
            min = np.sort(unmasked_weights)[0]
            max = np.sort(unmasked_weights)[-1]
            
            bits = opt.conv_bits ## bits to represent shared weights
            if c == 0:
                print("Unmasked weights: ", unmasked_weights)
            
            # cluster the unmasked weights into shared weights
            s_weights = np.linspace(min, max, 2**bits)
            kmeans = KMeans(n_clusters = len(s_weights), init = s_weights.reshape(-1,1), n_init=1)
            kmeans.fit(unmasked_weights.reshape(-1,1))
            for i, label in enumerate(kmeans.labels_):
                unmasked_weights[i] = kmeans.cluster_centers_[label][0]
            if c == 0:
                print("Shared weights", kmeans.cluster_centers_)
            if c == 0:
                print("Unmasked new weights: ", unmasked_weights)
            
            # copy new shared weights back to original weights where mask allows
            alt_index = 0
            for i, mask_val in enumerate(mask_flatten):
                if mask_val:
                    weights_flatten[i] = unmasked_weights[alt_index]
                    alt_index += 1
            
            # place weights in state dictionary
            weights = weights_flatten.reshape(weights.shape)
            weights = torch.from_numpy(weights)
            state_dict[weight_key] = weights
            c+=1
            
    model.load_state_dict(state_dict)
            

def global_shared_weights(model):
    state_dict = model.state_dict()
    all_weights = np.array([])

    for key, value in state_dict.items():
        if ("conv" in key and "weight" in key):
            copy_weights = value.cpu().numpy()
            copy_weights = copy_weights.flatten()
            all_weights = np.append(all_weights,copy_weights)
    min = np.amin(all_weights)
    max = np.amax(all_weights)
    print("Final length: ", len(all_weights))
    print("Min:", min)
    print("Max:", max)
    
    bits = 7
    s_weights = np.linspace(min, max, 2**bits)
    kmeans = KMeans(n_clusters = len(s_weights), init = s_weights.reshape(-1,1), n_init=1)
    kmeans.fit(all_weights.reshape(-1,1))
    for i, label in enumerate(kmeans.labels_):
        all_weights[i] = kmeans.cluster_centers_[label][0]
    print("Donezooooo")
    
    c_i = 0
    for key, value in state_dict.items():
        if ("conv" in key and "weight" in key):
            copy_weights = value.cpu().numpy()
            original_shape = copy_weights.shape
            copy_weights = copy_weights.flatten()
    

def share_weights(model):
    state_dict = model.state_dict()
    thresholds = determine_thresholds(model)
    for key, value in state_dict.items():
        if ("conv" in key and "weight" in key):
            print("key:", key)
            #get min and max value
            copy_weights = value.cpu().numpy()
            original_shape = copy_weights.shape
            copy_weights = copy_weights.flatten()

            weights_sorted = np.sort(copy_weights)
            min = weights_sorted[0]
            max = weights_sorted[-1]
            

            bits = get_bits(weights_sorted, thresholds)
            
            s_weights = np.linspace(min, max, 2**bits)
            kmeans = KMeans(n_clusters = len(s_weights), init = s_weights.reshape(-1,1), n_init=1)
            kmeans.fit(copy_weights.reshape(-1,1))
            for i, label in enumerate(kmeans.labels_):
                copy_weights[i] = kmeans.cluster_centers_[label][0]
            copy_weights = copy_weights.reshape(original_shape)
            copy_weights = torch.from_numpy(copy_weights)
            state_dict[key] = copy_weights


    model.load_state_dict(state_dict)
    return

def get_bits(weights_sorted, thresholds):
    # print("Range:", weights_sorted[-1] - weights_sorted[0])
    # print("Stdev:", np.std(weights_sorted))
    # print("Length:", len(weights_sorted))
    # print('\n')
    rnge = weights_sorted[-1] - weights_sorted[0]
    stdev = np.std(weights_sorted)
    length = len(weights_sorted)
    
    c_metric = rnge
    if c_metric <= thresholds[1]:
        return 4
    if c_metric <= thresholds[2]:
        return 5
    if c_metric <= thresholds[3]:
        return 6
    print("None Selected!!: ", c_metric, thresholds)
    # return opt.conv_bits
    
def determine_thresholds(model):
    state_dict = model.state_dict()
    stdev = []
    rnge = []
    num_layers = 0
    length = []
    for key, value in state_dict.items():
        if ("conv" in key and "weight" in key):
            copy_weights = value.cpu().numpy()
            copy_weights = copy_weights.flatten()
            copy_weights = np.sort(copy_weights)
            
            stdev.append(np.std(copy_weights))
            rnge.append(copy_weights[-1] - copy_weights[0])
            length.append(len(copy_weights))
            num_layers += 1
            
    stdev = np.sort(stdev)
    rnge = np.sort(rnge)
    length = np.sort(length)
    std_thresholds = [stdev[0], stdev[int(num_layers/3)], stdev[int(2*num_layers/3)], stdev[-1]]
    rnge_thresholds = [rnge[0], rnge[int(num_layers/3)], rnge[int(2*num_layers/3)], rnge[-1]]
    len_thresholds = [length[0], length[int(num_layers/3)], length[int(2*num_layers/3)], length[-1]]
    print("Std Thresholds:", std_thresholds)
    print("Rnge Thresholds:", rnge_thresholds)
    print("length thresholds:", len_thresholds)
    return rnge_thresholds
    
def determine_compression(model):
    thresholds = determine_thresholds(model)
    state_dict = model.state_dict()
    weights_encoded_by_bits = {4:0, 5:0, 6:0, 7:0, 8:0}
    layers_by_bits = {4:0, 5:0, 6:0, 7:0, 8:0}
    num_shared_weights = 0
    num_layers = 0
    num_params = 0
    for key, value in state_dict.items():
        if ("conv" in key and "weight" in key):
            copy_weights = value.cpu().numpy()
            copy_weights = copy_weights.flatten()
            
            # get the total number of weights in the network
            num_params += len(copy_weights)
            
            # get total number of shared weights in this layer
            bits = get_bits(np.sort(copy_weights), thresholds)
            num_shared_weights += 2**bits
            weights_encoded_by_bits[bits] += len(copy_weights)
            layers_by_bits[bits] += 1
            num_layers += 1
            
    print("Total number of weights:", num_params)
    print("Total number of shared weights:", num_shared_weights)
    print("Num of times different bits used:", layers_by_bits)
    print("Total number of weights encoded by bits:", weights_encoded_by_bits)
    sum = 0
    for key, value in weights_encoded_by_bits.items():
        sum+=value
    assert(sum == num_params)
    
    sum = 0
    for key, value in layers_by_bits.items():
        sum += value
    assert (sum == num_layers)
    
    sum = 0
    for key, value in weights_encoded_by_bits.items():
        sum += key*value
    
    cr = (num_params * 32)/ (num_shared_weights * 32 + sum)
    print("Compression Rate: ", cr)
    return
            
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, max_batches = 1000):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    start_time = time.time()
    batch_times = []

    num_batches = 0

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        batch_start_time = time.time()
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        batch_end_time = time.time()
        batch_times.append((batch_end_time - batch_start_time) / batch_size)

        num_batches+=1
        if num_batches >= max_batches:
            break


    # Concatenate sample statistics
    end_time = time.time()
    print("Total eval time: ", (end_time-start_time))
    print("Average inference time: ", (sum(batch_times) / len(batch_times)))
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--conv_bits", type=int, default = 7, help = "number of bits to represent conv shared weights")
    parser.add_argument("--pruned_model", type=bool, default = False, help = "Are you loading a pruned model?")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if (opt.pruned_model):
        model = torch.load("checkpoints/yolov3_pruned_model.pth")
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))


    #determine_stdev_thresholds(model)
    #determine_compression(model)

    
    if not opt.pruned_model:
        #global_shared_weights(model)
        determine_compression(model)
        share_weights(model)
    else:
        share_weights_pruned(model)
    
    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        max_batches = 1000,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

    
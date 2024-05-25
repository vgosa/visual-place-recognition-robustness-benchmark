
from datetime import datetime
import os
import re
import torch
import shutil
import logging
import torchscan
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA
from dataclasses import dataclass
from dataclass_csv import DataclassWriter
from csv_class import Result

import datasets_ws


def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    output = torchscan.utils.format_info(module_info)
    return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    if args.backbone == "transvpr":
        state_dict = OrderedDict({f'backbone.{k}': v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    # if (args.backbone == "transvpr"):
    #     patch_feat = model(input)
    #     global_feat, attention_mask = model.pool(patch_feat)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca

def convert_recalls_to_csv(recalls, args, corruption=None, severity=None):
    result = Result(
        timestamp= datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        backbone=args.backbone,
        aggregation=args.aggregation,
        checkpoint_path=args.resume,
        pca=args.pca_dim is not None,
        pca_dim=args.pca_dim,
        dataset=args.dataset_name,
        resize_H=args.resize[0],
        resize_W=args.resize[1],
        corruption=corruption,
        severity= severity,
        recall_1=recalls[0],
        recall_5=recalls[1],
        recall_10=recalls[2],
        recall_20=recalls[3]
    )
    return result

def save_csv_to_file(args, results):
    print(f"Saving this test's results to {os.path.join(args.save_dir, 'results.csv')}.")
    print("Appended all results to the general results.csv file in the current directory.")
    print("WARNING: This method only supports the default recall rates values. Any overwriting will result in a erroneous csv save.")
    # Save the results locally
    with open(os.path.join(args.save_dir, "results.csv"), 'a') as f:
        writer = DataclassWriter(f, results, Result)
        writer.write()
    # Save the results to a centralized csv file
    with open("results.csv", 'a') as f:
        writer = DataclassWriter(f, results, Result)
        if os.stat("results.csv").st_size == 0:
            writer.write()
        else:
            writer.write(skip_header=True)

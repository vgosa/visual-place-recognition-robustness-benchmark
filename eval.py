
"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.
The first is https://github.com/naver/deep-image-retrieval from Naver labs,
provides ResNet-50 and ResNet-101 trained with AP on Google Landmarks 18 clean.
$ python eval.py --off_the_shelf=naver --l2=none --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

The second is https://github.com/filipradenovic/cnnimageretrieval-pytorch from
Radenovic, provides ResNet-50 and ResNet-101 trained with a triplet loss
on Google Landmarks 18 and sfm120k.
$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048
$ python eval.py --off_the_shelf=radenovic_sfm --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

from collections import OrderedDict
import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd
from corruption import corruptions

import test
import util
import commons
import datasets_ws
from model import network
from util import save_csv_to_file

OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, "{backbone}_{dataset_name}".format(backbone = args.backbone, dataset_name = args.dataset_name), start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
if args.backbone.startswith("selavpr"):
    model = network.SelaVPRNet(args)
elif args.network == "cosplace":
    model = network.CosPlace(args)
elif args.backbone.startswith("dinov2"):
    model = network.DinoV2(args)
elif args.network == "cricavpr":
    args.features_dim = 14*768
    model = network.CricaVPR(args)
else:
    model = network.GeoLocalizationNet(args)

model = model.to(args.device)

if args.backbone.startswith("selavpr"):
    # model = torch.nn.DataParallel(model)
    if args.resume != None:
        state_dict = torch.load(args.resume)["model_state_dict"]
        if list(state_dict.keys())[0].startswith('module'):
            state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
        model.load_state_dict(state_dict)

    if args.pca_dim == None:
        pca = None
    else:
        full_features_dim = args.features_dim
        args.features_dim = args.pca_dim
        pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

if args.aggregation in ["netvlad", "crn"]:
    if args.network != 'cosplace' and args.backbone != "dinov2":
        args.features_dim *= args.netvlad_clusters

if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
    if args.off_the_shelf.startswith("radenovic"):
        pretrain_dataset_name = args.off_the_shelf.split("_")[1]  # sfm or gldv1 datasets
        url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
        state_dict = load_url(url, model_dir=join("data", "off_the_shelf_nets"))
    else:
        # This is a hacky workaround to maintain compatibility
        sys.modules['sklearn.decomposition.pca'] = sklearn.decomposition._pca
        zip_file_path = join("data", "off_the_shelf_nets", args.backbone + "_naver.zip")
        if not os.path.exists(zip_file_path):
            gdd.download_file_from_google_drive(file_id=OFF_THE_SHELF_NAVER[args.backbone],
                                                dest_path=zip_file_path, unzip=True)
        if args.backbone == "resnet50conv5":
            state_dict_filename = "Resnet50-AP-GeM.pt"
        elif args.backbone == "resnet101conv5":
            state_dict_filename = "Resnet-101-AP-GeM.pt"
        state_dict = torch.load(join("data", "off_the_shelf_nets", state_dict_filename))
    state_dict = state_dict["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)
elif args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)
    
    
########################################## TEST on TEST SET #########################################
def run_test(args, model, test_ds, pca, corruption=None, severity=None):
    start_time_test = datetime.now()
    recalls, recalls_str, result = test.test(args, test_ds, model, args.test_method, pca, corruption, severity)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")

    logging.info(f"Finished in {str(datetime.now() - start_time_test)[:-7]}")
    logging.info(f"Elapsed time:{str(datetime.now() - start_time)[:-7]}")
    return result


######################################### DATASETS AND TEST #########################################
if args.corruption is None:
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")
    result = run_test(args, model, test_ds, pca)
    save_csv_to_file(args, [result])
else:
    assert args.corruption == "all" or args.corruption in corruptions, f"Choose a valid corruption: {corruptions}"
    if args.corruption == "all":
        for corruption in corruptions:
            if corruption in ['rainy', 'day_to_night']:
                print(f"Testing corruption=[{corruption}]")
                test_ds = datasets_ws.CorruptedDataset(args=args,
                                                       corruption=corruption,
                                                       datasets_folder=args.datasets_folder,
                                                       dataset_name=args.dataset_name,
                                                       split="test",
                                                       severity=1)
                save_csv_to_file(args, [run_test(args, model, test_ds, pca, corruption, 1)])
                continue
            print(f"Testing corruption=[{corruption}] with all severity levels")
            for severity in range(1, 6):
                print(f"Testing corruption=[{corruption}] with severity={severity}")
                test_ds = datasets_ws.CorruptedDataset(args=args,
                                                       corruption=corruption,
                                                       datasets_folder=args.datasets_folder,
                                                       dataset_name=args.dataset_name,
                                                       split="test",
                                                       severity=severity)
                save_csv_to_file(args, [run_test(args, model, test_ds, pca, corruption, severity)])
    elif args.severity:
        if args.corruption in ['rainy', 'day_to_night']:
            print(f"Cannot test severity for corruption=[{args.corruption}]")
            sys.exit()
        print(f"Testing corruption=[{args.corruption}] with severity={args.severity}")
        test_ds = datasets_ws.CorruptedDataset(args=args,
                                               corruption=args.corruption,
                                               datasets_folder=args.datasets_folder,
                                               dataset_name=args.dataset_name,
                                               split="test",
                                               severity=args.severity)
        save_csv_to_file(args, [run_test(args, model, test_ds, pca, args.corruption, args.severity)])
    else:
        if args.corruption in ['rainy', 'day_to_night']:
            print(f"Testing corruption=[{args.corruption}]")
            test_ds = datasets_ws.CorruptedDataset(args=args,
                                                    corruption=args.corruption,
                                                    datasets_folder=args.datasets_folder,
                                                    dataset_name=args.dataset_name,
                                                    split="test",
                                                    severity=1)
            save_csv_to_file(args, [run_test(args, model, test_ds, pca, args.corruption, 1)])
            sys.exit()
        print(f"Testing corruption=[{args.corruption}] with all severity levels")
        for severity in range(1, 6):
            print(f"Testing corruption=[{args.corruption}] with severity={severity}")
            test_ds = datasets_ws.CorruptedDataset(args=args,
                                                   corruption=args.corruption,
                                                   datasets_folder=args.datasets_folder,
                                                   dataset_name=args.dataset_name,
                                                   split="test",
                                                   severity=severity)
            save_csv_to_file(args, [run_test(args, model, test_ds, pca, args.corruption, severity)])

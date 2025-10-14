import os
import gc
import json
import time
import time
import torch
import random
import psutil
import MyUtils
import MyDatasets
import MyBaselines
import numpy as np
import torchvision
import transformers
import tensorflow as tf
import matplotlib.pyplot as plt
from Config import get_arguments
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





##############################################################################################################
##############################################################################################################

def main(args):
    
    if args.setup == "local":
        avg_test_Acc = MyBaselines.run_local(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "fedavg":
        avg_test_Acc = MyBaselines.run_fedavg(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "fedmd_yn":
        avg_test_Acc = MyBaselines.run_fedmd(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "fedmd_mix_yn":
        avg_test_Acc = MyBaselines.run_fedmd_mix(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "zero_shot":
        avg_test_Acc = MyBaselines.run_zero_shot(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "open_vocab":
        avg_test_Acc = MyBaselines.run_open_vocab(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "fl_vocab":
        avg_test_Acc = MyBaselines.run_open_vocab(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "sidclip":
        avg_test_Acc = MyBaselines.run_sid_clip(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)

    elif args.setup == "koala":
        avg_test_Acc = MyBaselines.run_koala(distributed_dataset, num_classes, name_classes, synthetic_public_data, original_public_data, args)

    elif args.setup == "proposed_yn":
        avg_test_Acc = MyBaselines.run_proposed(distributed_dataset, num_classes, name_classes, synthetic_public_data, args)
    else:
        raise ValueError(f"Unknown setup: {args.setup}")

    MyUtils.save_as_json(avg_test_Acc, args, file_name= args.output_name + "accuracy_" + args.setup)

##############################################################################################################
##############################################################################################################

dataset_loaders = {
    "cifar10": MyDatasets.load_cifar10,
    "eurosat": MyDatasets.load_eurosat,
    "svhn": MyDatasets.load_svhn,
    "fashion_mnist": MyDatasets.load_fashion_mnist
}



if __name__ == "__main__":
    args = get_arguments()

    loader = dataset_loaders.get(args.dataset.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset, num_classes, name_classes, original_public_data = loader(args.num_train_samples, args.num_test_samples)


    distributed_dataset, num_samples = MyDatasets.data_distributing(dataset, num_classes, args.alpha_dirichlet, args.num_clients)
    print("\n ]data distribution of devices: \n", num_samples)


    synthetic_public_data = MyUtils.load_synthetic_images(name_classes, 
                                                  image_size = dataset["train"]["image"][0].shape[-2:], 
                                                  data_dir = "/project/def-arashmoh/shahab33/GenFKD/Synthetic_Image/CIFAR10",
                                                  max_per_class=args.num_synth_img_per_class)

    
    #synthetic_public_data = original_public_data
  

    MyUtils.set_seed(42)


    configurations = [
        {"setup": "proposed_yn"},
        {"setup": "local"},
        {"setup": "fedavg"},
        {"setup": "fedmd_yn"},
        #{"setup": "zero_shot"},
        #{"setup": "open_vocab"},
        #{"setup": "sidclip"},
        #{"setup": "koala"},
           
    ]


    for config in configurations:
        args.setup = config["setup"]

        separator = "=" * 40
        print(f"\n{separator} Running configuration: {args.setup} {separator}")

        main(args)

        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")
        MyUtils.clean_up_memory()

    MyUtils.load_and_plot_results("results", args.output_name)


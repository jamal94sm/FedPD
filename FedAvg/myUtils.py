import torch
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import json
from sklearn.metrics import accuracy_score
import gc
from torch.utils.data import DataLoader, TensorDataset



##############################################################################################################
##############################################################################################################

def Evaluate(model, images, labels, device, batch_size=64):
    model.eval()
    correct = 0
    all_preds = []

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            pred = model(batch_images)
            predicted_classes = torch.argmax(pred, dim=1)
            correct += (predicted_classes == batch_labels).sum().item()
            all_preds.append(pred.cpu())

    accuracy = 100.0 * correct / len(labels)
    return accuracy, torch.cat(all_preds, dim=0)

##############################################################################################################
############################################################################################################## 

def Evaluate2(ground_truth, output_logits):
    with torch.no_grad():
        predicted_classes = torch.argmax(output_logits, dim=1)
        accuracy = accuracy_score(
            ground_truth.cpu().numpy(),
            predicted_classes.cpu().numpy()
        )
    return accuracy

##############################################################################################################
############################################################################################################## 

def Train(model, data, optimizer, scheduler, loss_fn,  batch_size, epochs, device, debug):

    dataset = torch.utils.data.DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )


    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for batch in dataset:
            optimizer.zero_grad()
            pred = model( batch['image'].to(device) )
            error = loss_fn(pred, batch["label"].to(device))
            error.backward()
            optimizer.step()
            batch_loss.append(float(error))
        scheduler.step()
        epoch_loss.append(np.mean(batch_loss))
        epoch_acc.append( Evaluate(model,  data["train"]["image"], data["train"]["label"], device)[0] )

        if data['test'] is not None:
            epoch_test_acc.append( Evaluate(model,  data["test"]["image"], data["test"]["label"], device)[0] )
        
        if debug: print("Epoch {}/{} ===> Loss: {:.2f}, Train accuracy: {:.2f}, Test accuracy: {:.2f}".format(epoch, epochs, epoch_loss[-1], epoch_acc[-1], epoch_test_acc[-1]))
    

    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    
    
    return epoch_loss, epoch_acc, epoch_test_acc

##############################################################################################################     
##############################################################################################################

def model_size(model): # returns if Giga Bytes
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = total_params * 4  # 4 bytes per parameter (float32)
    size_gB = size_bytes / (1024 ** 3)
    return size_gB

##############################################################################################################
##############################################################################################################

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

def estimate_flops(model, dataset_dict, criterion, optimizer, device, batch_size=4):
    model.to(device)
    model.train()


    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, hf_data):
            self.data = hf_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]['image']
            label = self.data[idx]['label']
            return image, label

    dataloader = DataLoader(TensorDataset(dataset_dict['train']), batch_size=batch_size)

    # Get input shape from one batch
    inputs, _ = next(iter(dataloader))
    input_size = tuple(inputs.shape)

    # Estimate FLOPs using torchinfo
    info = summary(model, input_size=input_size, verbose=0)
    flops_per_forward = 2 * info.total_mult_adds

    num_batches = len(dataloader)
    flops_per_batch = flops_per_forward * 2  # forward + backward
    total_flops = flops_per_batch * num_batches

    total_gflops = total_flops / 1e9  # Convert to GFLOPs
    #print("====> Total FLOPs: {:.2f} GFLOPs".format(total_gflops))
    return total_gflops


##############################################################################################################
##############################################################################################################

import torch
import numpy as np

def get_dataset_size_in_gb(dataset_dict):
    def estimate_split_size(split):
        total_bytes = 0
        for sample in split:
            image = sample['image']
            label = sample['label']
            if isinstance(image, torch.Tensor):
                total_bytes += image.numel() * image.element_size()
            elif isinstance(image, np.ndarray):
                total_bytes += image.nbytes
            elif hasattr(image, 'tobytes'):
                total_bytes += len(image.tobytes())
            total_bytes += 4  # Assuming label is int32
        return total_bytes

    total_bytes = 0
    if 'train' in dataset_dict:
        total_bytes += estimate_split_size(dataset_dict['train'])
    if 'test' in dataset_dict and dataset_dict['test'] is not None:
        total_bytes += estimate_split_size(dataset_dict['test'])

    size_gb = total_bytes / (1024 ** 3)
    return size_gb


##############################################################################################################
##############################################################################################################
def get_logits_size_in_gb(num_samples, num_classes):
    total_elements = num_samples * num_classes
    total_bytes = total_elements * 4  # float32 = 4 bytes
    size_gb = total_bytes / (1024 ** 3)
    return size_gb

############################################################################################################## 
############################################################################################################## 
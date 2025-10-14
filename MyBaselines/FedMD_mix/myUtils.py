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

def extend_proto_outputs_to_labels(input_data, proto_outputs):
    num_data = input_data["train"]["image"].shape[0]
    num_classes = len(  sorted(set(input_data["train"]["label"].tolist()))  )
    labels = input_data["train"]["label"]
    extended_outputs = torch.zeros(num_data, num_classes)
    for i in range(num_data):
        extended_outputs[i] = proto_outputs[labels[i].item()]
    return extended_outputs

##############################################################################################################
##############################################################################################################

def Distill_mix(model, extended_data, data, optimizer, loss_fn, scheduler, batch_size, epochs, device):
    epoch_loss = []
    epoch_acc = []
    epoch_test_acc = []


    model.train()
    temperature = 1
    alpha = 1 # weight for distillation loss

    dataset = torch.utils.data.DataLoader(
        extended_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,              # Enables multi-processing
        pin_memory=True,            # Speeds up host-to-GPU transfer
        prefetch_factor=2,          # Controls preloading per worker
        persistent_workers=False    # Keeps workers alive between epochs
    )

    for epoch in range(epochs):
        batch_loss = []
        model.train()
        for batch in dataset:
            optimizer.zero_grad()

            local_preds = model(batch["student_model_local_input"].to(device))
            public_preds = model(batch["student_model_public_input"].to(device))

            error1 = loss_fn(local_preds, batch["student_model_local_output"].to(device))
            error2 = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(public_preds / temperature, dim=1),
                torch.nn.functional.softmax( batch["teacher_knowledge"].to(device) / temperature, dim=1),
                reduction='batchmean')

            
            error = error1 + error2
            error.backward()
            optimizer.step()
            batch_loss.append(float(error))
        scheduler.step()
        epoch_loss.append(np.mean(batch_loss))

        if data:
            epoch_acc.append(Evaluate(model, data["train"]["image"], data["train"]["label"], device)[0])
            epoch_test_acc.append(Evaluate(model, data["test"]["image"], data["test"]["label"], device)[0])

    del dataset
    gc.collect()
    torch.cuda.empty_cache() 

    return epoch_loss, epoch_acc, epoch_test_acc



##############################################################################################################
##############################################################################################################

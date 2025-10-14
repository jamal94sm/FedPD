import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import datasets
from datasets import DatasetDict, concatenate_datasets
from . import myModels
from . import myUtils



##############################################################################################################
##############################################################################################################

import torch
from torch.utils.data import Dataset
import random

class DistillDataset(Dataset):
    def __init__(self, local_images, local_labels, public_images, teacher_outputs):
        self.local_images = local_images
        self.local_labels = local_labels
        self.public_images = public_images
        self.teacher_outputs = teacher_outputs

        self.local_len = len(local_images)
        self.public_len = len(public_images)

    def __len__(self):
        # You can choose to iterate based on the smaller or larger dataset
        return max(self.local_len, self.public_len)

    def __getitem__(self, idx):
        # Sample local data
        local_idx = idx % self.local_len
        local_image = self.local_images[local_idx]
        local_label = self.local_labels[local_idx]

        # Sample public data
        public_idx = idx % self.public_len
        public_image = self.public_images[public_idx]
        teacher_output = self.teacher_outputs[public_idx]

        return {
            "student_model_local_input": local_image,
            "student_model_local_output": local_label,
            "student_model_public_input": public_image,
            "teacher_knowledge": teacher_output
        }

##############################################################################################################
##############################################################################################################
class Server():

    def __init__(self, clients):
        self.clients = clients
    
    def aggregation(self):
        summ = torch.stack([client.logits for client in self.clients]).sum(dim=0)
        self.ave_logits = summ / len(self.clients)
        return self.ave_logits


##############################################################################################################
##############################################################################################################
class Device():

    def __init__(self, ID, data, num_classes, name_classes, public_data, args):
        self.ID = ID
        self.data = data
        self.args = args
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.num_samples = torch.bincount(self.data["train"]["label"], minlength=num_classes)
        self.public_data = public_data

        
        if args.local_model_name=="MLP": #MLP
            self.model = myModels.MLP(data["train"]["image"].view(data["train"]["image"].size(0), -1).size(1), self.num_classes).to(args.device)
        elif args.local_model_name=="ResNet": 
            self.model = myModels.ResNet([1, 1, 1], self.num_classes).to(args.device) #ResNet
        elif args.local_model_name=="CNN": 
            self.model = myModels.LightWeight_CNN(data["train"]["image"][0].shape, self.num_classes, 3).to(args.device) #CNN
        elif args.local_model_name=="MobileNetV2":
            self.model = myModels.MobileNetV2(data["train"]["image"][0].shape, self.num_classes).to(args.device) #MobileNetV2
        elif args.local_model_name=="ResNet18":
            self.model = myModels.ResNet18(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet18
        elif args.local_model_name=="ResNet10":
            self.model = myModels.ResNet10(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet10
        elif args.local_model_name=="ResNet20":
            self.model = myModels.ResNet20(data["train"]["image"][0].shape, self.num_classes).to(args.device) #ResNet20
        elif args.local_model_name=="EfficientNet":
            self.model = myModels.EfficientNet(data["train"]["image"][0].shape, self.num_classes).to(args.device) #EfficientNet



        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        self.Acc = []
        self.test_Acc = []


        #self.GB_model_size = MyUtils.model_size(self.model)
        #self.G_flops = MyUtils.estimate_flops(self.model, self.data, self.loss_fn, self.optimizer, args.device)

    def local_distillation_mix(self, teacher_knowledge):

        teacher_knowledge = teacher_knowledge.cpu()

        dataset = DistillDataset(
            local_images=self.data["train"]["image"],
            local_labels=self.data["train"]["label"],
            public_images=self.public_data["train"]["image"],
            teacher_outputs=teacher_knowledge
        )

        a, b, c = myUtils.Distill_mix(
            model=self.model,
            extended_data=dataset,
            data = self.data,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            scheduler=self.scheduler,
            batch_size=self.args.local_batch_size,
            epochs=self.args.local_epochs,
            device=self.args.device
        )

        self.Loss += a
        self.Acc += b
        self.test_Acc += c

    def ddf(self, x):
        x = datasets.Dataset.from_dict(x)
        x.set_format("torch")
        return x

    def cal_logits(self, data, proto=False, sifting=False):
        images = data["train"]["image"]
        labels = data["train"]["label"]

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=64)

        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.args.device)
                batch_labels = batch_labels.to(self.args.device)
                logits = self.model(batch_images)
                all_logits.append(logits)
                all_labels.append(batch_labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        unique_classes = sorted(set(labels.tolist()))
        num_classes = len(unique_classes)

        self.logits = logits


##############################################################################################################
##############################################################################################################




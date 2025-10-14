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
class Server():

    def __init__(self, clients, model, public_data, args):
        self.clients = clients
        self.args = args
        self.model = model
        self.clients = clients
        self.public_data = public_data
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        
    def ddf(self, x):
        x = datasets.Dataset.from_dict(x)
        x.set_format("torch")
        return x
    
    def fedavg_aggregation(self):
        import copy
        import torch

        Models = [client.model for client in self.clients]
        global_dict = copy.deepcopy(Models[0].state_dict())

        # Initialize global_dict to zeros
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])

        # Sum all local model parameters
        for model in Models:
            local_dict = model.state_dict()
            for key in global_dict:
                global_dict[key] += local_dict[key]

        # Average the parameters
        for key in global_dict:
            global_dict[key] = global_dict[key] / len(Models)

        # Create a new model and load the averaged state_dict
        global_model = copy.deepcopy(Models[0])  # assumes all models share the same architecture
        global_model.load_state_dict(global_dict)

        return global_model

    def get_general_knowledge(self):
        with torch.no_grad():
            pred = self.model(self.public_data["train"]["image"].to(self.args.device), inference=True)
        return pred

    def distill_generator(self, data, logits):
        teacher_knowledge = logits
        
        #data_for_extension = { "train": {"image": data["train"]["image"], "label": data["train"]["label"] } }
        #teacher_knowledge = myUtils.extend_proto_outputs_to_labels(data_for_extension, teacher_knowledge)
        
        extended_data = self.ddf({
            "student_model_input":  data["train"]["image"],
            "student_model_output": data["train"]["label"],
            "teacher_knowledge": teacher_knowledge
        })

        loss, _, _ = myUtils.Distil(
            model = self.model,
            extended_data = extended_data,
            data = None,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            loss_fn = self.loss_fn,
            batch_size = self.args.global_batch_size if "M" in self.args.setup else 8,
            epochs = self.args.global_epochs,
            device = self.args.device,
            debug = self.args.debug,
            args = self.args
        )
        self.Loss += loss

##############################################################################################################
##############################################################################################################
class Device():

    def __init__(self, ID, data, num_classes, name_classes, public_data, args):
        self.ID = ID
        self.data = data
        self.args = args

        self.num_classes = num_classes
        self.name_classes = name_classes
        self.num_samples = torch.bincount(self.data["train"]["label"], minlength=num_classes).to(args.device)
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


        #self.GB_model_size = myUtils.model_size(self.model)
        #self.G_flops = myUtils.estimate_flops(self.model, self.data, self.loss_fn, self.optimizer, args.device)

    def ddf(self, x):
        x = datasets.Dataset.from_dict(x)
        x.set_format("torch")
        return x

    def local_distillation(self, data, teacher_knowledge, proto=False):
        if proto:
            teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(data, teacher_knowledge)


        min_len = min(
            len(data["train"]["image"]),
            len(data["train"]["label"]),
            len(teacher_knowledge)
        )

        extended_data = self.ddf({
            "student_model_input": data["train"]["image"][:min_len],
            "student_model_output": data["train"]["label"][:min_len],
            "teacher_knowledge": teacher_knowledge[:min_len]
        })


        
        a, b, c = myUtils.Distil(self.model, extended_data, self.data, self.optimizer, self.scheduler, self.loss_fn,
                                 self.args.local_batch_size, self.args.local_epochs, self.args.device, self.args.debug, self.args)

    def local_training(self, data):
        a,b, c = myUtils.Train(self.model, data, self.optimizer, self.scheduler, self.loss_fn,
                               self.args.local_batch_size, self.args.local_epochs, self.args.device, self.args.debug)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c


##############################################################################################################
##############################################################################################################




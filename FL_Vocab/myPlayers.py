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

    def zero_shot(self, data, FM, processor, tokenizer, proto=False, batch_size=16):
        
        device = self.args.device
        FM = FM.to(device)

        processor.image_processor.do_rescale = False
        processor.image_processor.do_normalize = False

        images = data["image"]
        labels = data["label"]
        
        
        img_reps = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            inputs = processor(
                text=["blank"] * len(batch_images),
                images=batch_images,
                return_tensors="pt"
                )

            with torch.no_grad():
                pixel_values = inputs["pixel_values"].to(device)
                batch_img_rep = FM.get_image_features(pixel_values)
                batch_img_rep = batch_img_rep / batch_img_rep.norm(p=2, dim=-1, keepdim=True)
                img_reps.append(batch_img_rep.cpu())  # Move back to CPU to save GPU memory

            del inputs, pixel_values, batch_img_rep
            torch.cuda.empty_cache()

        img_rep = torch.cat(img_reps, dim=0).to(device)

        with torch.no_grad():
            text_rep = self.model.basic_text_rep / self.model.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * img_rep @ text_rep.t()

        if not proto:
            return logits

        unique_classes = sorted(set(labels.tolist()))
        num_classes = len(unique_classes)
        proto_logits = torch.empty((num_classes, num_classes), device=logits.device)

        for c in unique_classes:
            mask = (labels == c)
            category_logits = logits[mask].mean(dim=0)
            proto_logits[c] = category_logits

        return proto_logits

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


        #self.GB_model_size = MyUtils.model_size(self.model)
        #self.G_flops = MyUtils.estimate_flops(self.model, self.data, self.loss_fn, self.optimizer, args.device)

    def ddf(self, x):
        x = datasets.Dataset.from_dict(x)
        x.set_format("torch")
        return x

    def local_distillation(self, data, teacher_knowledge, proto=False):
        if proto:
            teacher_knowledge = myUtils.extend_proto_outputs_to_labels(data, teacher_knowledge)

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
                                 self.args.local_batch_size, self.args.local_epochs, self.args.device, self.args.debug)


##############################################################################################################
##############################################################################################################




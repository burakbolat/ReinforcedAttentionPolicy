import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import model

class Trainer:
    def __init__(self, config):
        self.num_classes = config["num_classes"]
        self.transform = config["transform"]
        # self.train_set = config["train_set_dir"]
        # self.test_set = config["test_set_dir"]
        self.batch_size = config["batch_size"]
        self.finetune_epoch = config["finetune_epoch"]

        use_pretrained = config["use_pretrained"]
        backbone_dir = config["backbone_dir"]
        self.device = config["device"]

        if use_pretrained: self.backbone = model.resnet18(True, load_path=backbone_dir, num_classes = self.num_classes)
        else: self.backbone = model.resnet18(False, num_classes = self.num_classes)
        self.backbone = self.backbone.to(self.device)

        self.train_set = torchvision.datasets.CIFAR10("CIFAR10", True, transform=transform, download=True)
        self.test_set = torchvision.datasets.CIFAR10("CIFAR10", False, transform=transform, download=True) 
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)
    
    def set_parameter_requires_grad(self, model, feature_extracting): 
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def finetune_backbone(self, backbone, feature_extracting=True, first_loop=False):
        # Followed https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        
        if first_loop:
            self.set_parameter_requires_grad(backbone, feature_extracting=feature_extracting)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, self.num_classes)
            backbone.fc = backbone.fc.to(self.device)
            print("Changed last layer.")
            for name,param in backbone.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        optim = torch.optim.SGD(backbone.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True)
        # optim = torch.optim.Adam(backbone.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print("Starting training")
        for epoch in range(self.finetune_epoch):
            backbone.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                optim.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = backbone(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            
            print('{} Loss at {}: {:.4f}'.format("Train", epoch, epoch_loss))
            if epoch % 5 == 0: torch.save(backbone.state_dict(), "models/finetuned_resnet18_{}.pt".format(epoch))
    
    def eval(self, test_loader, model):
        correct = 0
        for batch, labels in test_loader:
            labels = labels.to(self.device)
            output = model(batch.float().to(self.device))
            predicted_label = torch.argmax(output, dim=1)
            correct += torch.sum(predicted_label == labels)

        return correct/len(test_loader.dataset)
        
if __name__=="__main__":
    # For torch pretrained models mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    config = {
        "transform" : transform,
        "num_classes" : 10,
        "batch_size" : 128,
        "finetune_epoch" : 100,
        "use_pretrained" : True,
        "backbone_dir" : "models/finetuned_resnet18_5_0.pt",
        "device" : "cuda"
    }

    trainer = Trainer(config=config)

    performance = trainer.eval(trainer.test_loader, trainer.backbone)
    print(performance)
    first_loop = True
    while performance < 92.6:
        print(performance)
        trainer.finetune_backbone(trainer.backbone, False, first_loop=first_loop)
        performance = trainer.eval(trainer.test_loader, trainer.backbone)
        first_loop = False
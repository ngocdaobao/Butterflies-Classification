from torchvision.models import alexnet
import torch.nn as nn
import torch
from sklearn.svm import SVC

#Keep the feature extractor fixed, change output into 100 classes

class alexnet_model(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, full_finetune=True):
        super(alexnet_model, self).__init__()
        self.model = alexnet(pretrained=pretrained)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        # Set feature extractor layers to be trainable if full_finetune:
        if full_finetune:
            for param in self.model.features.parameters():
                param.requires_grad = True
        else:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Set the classifier layers to be trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
""""alexnet implement from scratch
class alexnet_model_scratch(nn.Module):
    def __init__(self, num_classes=100):
        super(alexnet_model_scratch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
"""  


"""	
class SVM(nn.Module):
    def __init__(self, kernel='alexnet'):
        super(SVM, self).__init__()
        if kernel == 'alexnet':
            alex = alexnet(pretrained=True)
            self.feature_extractor = nn.Sequential(
                alex.features,
                alex.avgpool,  # Add avgpool layer to match the output size
                nn.Flatten()  # Remove the last layer
            )  # Remove the last layer
            self.projection = alex.classifier[:-1] 
        svm_head = nn.Linear(4096, 100)
        self.model = nn.Sequential(self.feature_extractor, self.projection, svm_head)
        # Freeze the feature extractor layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Set the classifier layers to be trainable
        for param in svm_head.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.model(x)
        return x
"""


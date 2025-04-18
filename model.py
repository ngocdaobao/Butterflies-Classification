from torchvision.models import alexnet
import torch.nn as nn
import torch

#Keep the feature extractor fixed, change output into 100 classes

class alexnet_model(nn.Module):
    def __init__(self, num_classes=100):
        super(alexnet_model, self).__init__()
        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        # Freeze the feature extractor layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Set the classifier layers to be trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class SVM(nn.Module):
    def __init__(self, input_dim, output_dim, kernel='linear'):
        super(SVM, self).__init__()
        self.kernel = kernel
        self.model = nn.Linear(input_dim, output_dim)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

from torchvision.models import alexnet
import torch.nn as nn
import torch
from sklearn.svm import SVC

#Keep the feature extractor fixed, change output into 100 classes

class alexnet_model(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(alexnet_model, self).__init__()
        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        # Freeze the feature extractor layers if pretrained is True
        if pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False
        else:
            for param in self.model.features.parameters():
                param.requires_grad = True
        # Set the classifier layers to be trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class SVM:
    def __init__(self, kernel='linear'):
        self.kernel = kernel
        self.svm = SVC(kernel=self.kernel, probability=True)

    def predict(self, X_train, y_train, X_test, y_test, X_valid, y_valid):
        self.svm.fit(X_train, y_train)
        predict = self.svm.predict(X_test)
        acc = (predict == y_test).sum()/len(y_test)
        return acc

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


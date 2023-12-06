# This code is written by Jingyuan Yang @ XD

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class model_baseline(nn.Module):
    """ResNet50 for Visual Sentiment Analysis on FI_8"""

    def __init__(self, base_model):
        super(model_baseline, self).__init__()
        self.fcn = nn.Sequential(*list(base_model.children())[:-2]) ##-2
        self.GAvgPool = nn.AvgPool2d(kernel_size=14)

        # classifier
        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier3 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier4 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier5 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier6 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier7 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        self.classifier8 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=8)
        )  # vgg19 512 resnet50 2048
        # self.classifier9 = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=1024, out_features=8)
        # )  # vgg19 512 resnet50 2048
        # self.classifier10 = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=1024, out_features=8)
        # )  # vgg19 512 resnet50 2048
        # self.classifier11 = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=1024, out_features=8)
        # )  # vgg19 512 resnet50 2048

    def forward(self, x):
        x = self.fcn(x)
        x = self.GAvgPool(x)
        x = x.view(x.size(0), x.size(1))

        #-------classifier8--------#
        emotion1 = self.classifier1(x)
        emotion1 = F.softmax(emotion1, dim=1)
        emotion1 = emotion1.unsqueeze(2)
        emotion2 = self.classifier2(x)
        emotion2 = F.softmax(emotion2, dim=1)
        emotion2 = emotion2.unsqueeze(2)
        emotion3 = self.classifier3(x)
        emotion3 = F.softmax(emotion3, dim=1)
        emotion3 = emotion3.unsqueeze(2)
        emotion4 = self.classifier4(x)
        emotion4 = F.softmax(emotion4, dim=1)
        emotion4 = emotion4.unsqueeze(2)
        emotion5 = self.classifier5(x)
        emotion5 = F.softmax(emotion5, dim=1)
        emotion5 = emotion5.unsqueeze(2)
        emotion6 = self.classifier6(x)
        emotion6 = F.softmax(emotion6, dim=1)
        emotion6 = emotion6.unsqueeze(2)
        emotion7 = self.classifier7(x)
        emotion7 = F.softmax(emotion7, dim=1)
        emotion7 = emotion7.unsqueeze(2)
        emotion8 = self.classifier8(x)
        emotion8 = F.softmax(emotion8, dim=1)
        emotion8 = emotion8.unsqueeze(2)
        # emotion9 = self.classifier9(x)
        # emotion9 = F.softmax(emotion9, dim=1)
        # emotion9 = emotion9.unsqueeze(2)
        # emotion10 = self.classifier10(x)
        # emotion10 = F.softmax(emotion10, dim=1)
        # emotion10 = emotion10.unsqueeze(2)
        # emotion11 = self.classifier11(x)
        # emotion11 = F.softmax(emotion11, dim=1)
        # emotion11 = emotion11.unsqueeze(2)

        # emotion_sum = (emotion1 + emotion2 + emotion3 + emotion4 + emotion5 + emotion6 + emotion7 + emotion8 + emotion9 + emotion10 + emotion11) / 11
        emotion_sum = (emotion1 + emotion2 + emotion3 + emotion4 + emotion5 + emotion6 + emotion7 + emotion8) / 8

        emotion_sum = emotion_sum.squeeze()
        # emotion_single = torch.cat([emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8, emotion9, emotion10, emotion11], dim=2)
        emotion_single = torch.cat([emotion1, emotion2, emotion3, emotion4, emotion5, emotion6, emotion7, emotion8], dim=2)

        return emotion_sum, emotion_single
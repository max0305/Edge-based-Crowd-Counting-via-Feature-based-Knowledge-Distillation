import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureConnector(nn.Module):
    """
    Projects Student features to match Teacher feature dimensions.
    Handles both Channel alignment (1x1 Conv) and Spatial alignment (Upsampling if needed).
    """
    def __init__(self, student_channels, teacher_channels):
        super(FeatureConnector, self).__init__()
        self.projector = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, student_features, teacher_spatial_size=None):
        x = self.projector(student_features)
        x = self.relu(x)
        
        # If spatial dimensions don't match, interpolate
        if teacher_spatial_size is not None and x.shape[2:] != teacher_spatial_size:
            x = F.interpolate(x, size=teacher_spatial_size, mode='bilinear', align_corners=False)
            
        return x

class DistillationModel(nn.Module):
    def __init__(self, teacher, student, connector):
        super(DistillationModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.connector = connector
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_output, teacher_features = self.teacher(x)
            
        # Student forward
        student_output, student_features = self.student(x)
        
        # Project student features
        projected_student_features = self.connector(student_features, teacher_features.shape[2:])
        
        return student_output, projected_student_features, teacher_features

def distillation_loss(student_output, target, projected_features, teacher_features, lambda_kd=100):
    # 1. Task Loss (MSE between Student Output and Ground Truth)
    task_loss = nn.MSELoss()(student_output, target)
    
    # 2. Distillation Loss (MSE between Projected Student Features and Teacher Features)
    # Hint Learning: "Where to look"
    dist_loss = nn.MSELoss()(projected_features, teacher_features)
    
    total_loss = task_loss + lambda_kd * dist_loss
    return total_loss, task_loss, dist_loss

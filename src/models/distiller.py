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
            x = F.interpolate(x, size=teacher_spatial_size, mode='bicubic', align_corners=False)
            
        return x

class DistillationModel(nn.Module):
    def __init__(self, teacher, student, connectors):
        super(DistillationModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.connectors = connectors # Should be nn.ModuleList
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def train(self, mode=True):
        """
        Override train mode to ensure teacher is ALWAYS in eval mode.
        Even when the student is training, the teacher must remain in eval mode
        to keep its BatchNorm statistics fixed.
        """
        super(DistillationModel, self).train(mode)
        self.teacher.eval()
        return self

    def forward(self, x):
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_output, teacher_features = self.teacher(x)
            
        # Student forward
        student_output, student_features = self.student(x)
        
        # Project student features
        projected_features = []
        for i, connector in enumerate(self.connectors):
            # teacher_features is a list now
            proj = connector(student_features[i], teacher_features[i].shape[2:])
            projected_features.append(proj)
        
        return student_output, projected_features, teacher_features

def distillation_loss(student_output, target, projected_features, teacher_features, lambda_kd=0.1):
    # Task Loss (MSE between Student Output and Ground Truth)
    task_loss = nn.MSELoss(reduction='sum')(student_output, target)
    
    # Distillation Loss (MSE between Projected Student Features and Teacher Features)
    dist_loss = 0.0
    for proj, teacher in zip(projected_features, teacher_features):
        dist_loss += nn.MSELoss()(proj, teacher)
    
    # Use additive combination: Task + Lambda * Dist
    # This matches Config.DISTILLATION_LAMBDA = 100.0
    total_loss = task_loss + lambda_kd * dist_loss
    return total_loss, task_loss, dist_loss

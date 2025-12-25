
class Config:
    # Project Metadata
    PROJECT_TITLE = "Edge-based Crowd Counting via Feature-based Knowledge Distillation"
    
    # Dataset
    DATASET_PATH = "data/shanghaitech_with_people_density_map/ShanghaiTech/part_B"
    INPUT_SIZE = (512, 512) # Example fixed size for training, or None for variable
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    
    # Model
    TEACHER_MODEL = "CSRNet"
    STUDENT_MODEL = "CrowdResNet18"
    
    # Distillation
    DISTILLATION_LAMBDA = 100 # Weight for feature loss
    TEACHER_CHANNELS = 512 # VGG16 backend channels
    STUDENT_CHANNELS = 256  # ResNet18 layer3 output channels
    
    # Training
    TEACHER_LR = 1e-5
    STUDENT_LR = 1e-4
    EPOCHS = 120
    DEVICE = "cuda"

    RANDOM_SEED = 1234


class Config:
    # Project Metadata
    PROJECT_TITLE = "Edge-based Crowd Counting via Feature-based Knowledge Distillation"
    
    # Dataset
    DATASET_PATH = "data/shanghaitech_with_people_density_map/ShanghaiTech/part_B"
    INPUT_SIZE = (512, 512) # Example fixed size for training, or None for variable
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    
    # Model
    TEACHER_MODEL = "CSRNet"
    STUDENT_MODEL = "Mobile-CSRNet"
    
    # Distillation
    DISTILLATION_LAMBDA = 100.0 # Weight for feature loss
    TEACHER_CHANNELS = 512 # VGG16 backend channels
    STUDENT_CHANNELS = 96  # MobileNetV2 backend channels (example)
    
    # Training
    LR = 1e-5
    EPOCHS = 200
    DEVICE = "cuda"

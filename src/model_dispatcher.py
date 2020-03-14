import models

MODEL_DISPATCHER = {
    'resnet18': models.resnet18,
    'efficientnet': models.efficientnet,
    'resnet50': models.resnet50
}
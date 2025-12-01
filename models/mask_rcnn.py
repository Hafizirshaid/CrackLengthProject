import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn


def get_model_instance_segmentation(num_classes: int):
    """
    Returns a Mask R-CNN model with a ResNet-50-FPN backbone.
    num_classes includes background (class=0).
    """
    # Load a model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor (classification + bbox regression)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model

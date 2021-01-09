from .mobilefacenet import MobileFaceNet
from .resnet import ResNet50, ResNet101


def create_backbone(net_type):
    net_creator = {'mobile': MobileFaceNet, 'r50': ResNet50, 'r101': ResNet101}
    if net_type not in net_creator:
        raise Exception('%s is not support in current implementation.' % net_type)
    return net_creator[net_type]()

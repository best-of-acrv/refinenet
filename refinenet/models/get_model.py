from models.refinenet import refinenet50, refinenet101, refinenet152
from models.refinenet_lw import refinenet_lw50, refinenet_lw101, refinenet_lw152


def get_model(args, pretrained='imagenet'):

    if args.dataset.lower() == 'nyu':
        num_classes = 40
    elif args.dataset.lower() == 'voc':
        num_classes = 21
    elif args.dataset.lower() == 'citiscapes':
        num_classes = 19
    else:
        print(
            'Invalid dataset chosen. Please choose from [nyu, voc, citiscapes]'
        )
        exit()

    if args.model_type.lower() == 'refinenet':

        if args.num_resnet_layers == 50:
            model = refinenet50(num_classes=num_classes, pretrained=pretrained)
        elif args.num_resnet_layers == 101:
            model = refinenet101(num_classes=num_classes,
                                 pretrained=pretrained)
        elif args.num_resnet_layers == 152:
            model = refinenet152(num_classes=num_classes,
                                 pretrained=pretrained)
        else:
            print(
                'Invalid number of ResNet layers chosen. Please choose from 50, 101 or 152 layers'
            )
            exit()

    elif args.model_type.lower() == 'refinenetlw':
        if args.num_resnet_layers == 50:
            model = refinenet_lw50(num_classes=num_classes,
                                   pretrained=pretrained)
        elif args.num_resnet_layers == 101:
            model = refinenet_lw101(num_classes=num_classes,
                                    pretrained=pretrained)
        elif args.num_resnet_layers == 152:
            model = refinenet_lw152(num_classes=num_classes,
                                    pretrained=pretrained)
        else:
            print(
                'Invalid number of ResNet layers chosen. Please choose from 50, 101 or 152 layers'
            )
            exit()

    else:
        print(
            'Invalid model type chosen. Please choose from [refinenet, refinenetlw]'
        )
        exit()

    return model

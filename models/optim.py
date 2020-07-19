import torch

def get_optimiser(args, model):
    if args.optimiser_type == 'adam':
        optimiser = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    elif args.optimiser_type == 'sgd':
        optimiser = torch.optim.SGD(lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, params=model.parameters())

    model.optimiser = optimiser
    return model

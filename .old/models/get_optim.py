import re
import torch

def get_encoder_and_decoder_params(model):
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            enc_params.append(v)
        else:
            dec_params.append(v)
    return enc_params, dec_params

def get_optimisers(args, model):

    # get parameters for encoder and decoder parts of model
    enc_params, dec_params = get_encoder_and_decoder_params(model)

    # RefineNet and RefineNet-LW have different hyperparameters for optimisers
    if args.model_type.lower() == 'refinenet':
        if args.optimiser_type.lower() == 'adam':
            enc_optimiser = torch.optim.Adam(lr=args.learning_rate, params=enc_params)
            dec_optimiser = torch.optim.Adam(lr=args.learning_rate, params=dec_params)
        elif args.optimiser_type.lower() == 'sgd':
            enc_optimiser = torch.optim.SGD(lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, params=enc_params)
            dec_optimiser = torch.optim.SGD(lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, params=dec_params)
    elif args.model_type.lower() == 'refinenetlw':
        if args.optimiser_type.lower() == 'adam':
            enc_optimiser = torch.optim.Adam(lr=args.learning_rate, params=enc_params)
            dec_optimiser = torch.optim.Adam(lr=10*args.learning_rate, params=dec_params)
        elif args.optimiser_type.lower() == 'sgd':
            enc_optimiser = torch.optim.SGD(lr=args.learning_rate, momentum=0.9, weight_decay=1e-5, params=enc_params)
            dec_optimiser = torch.optim.SGD(lr=10*args.learning_rate, momentum=0.9, weight_decay=1e-5, params=dec_params)
    else:
        print("Invalid model type selected. Please choose from: [refinenet, refinenetlw]")
        exit()

    # attach optimisers and schedulers to model
    model.enc_optimiser = enc_optimiser
    model.dec_optimiser = dec_optimiser
    return model



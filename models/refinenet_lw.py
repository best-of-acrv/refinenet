import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.blocks import *
from helpers.download_helper import download_model
from helpers.model_helper import find_snapshot
from utils.miou import compute_cm, compute_iu

class RefineNetLW(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(RefineNetLW, self).__init__()

        # check for cuda availability
        self.cuda_available = True if torch.cuda.is_available() else False

        # general params
        self.num_classes = num_classes
        self.inplanes = 64

        # resnet backbone
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # RefineNet
        # RefineNet Block 1
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)

        # RefineNet Block 2
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        # RefineNet Block 3
        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        # RefineNet Block 4
        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [ChainedResidualPoolLW(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)

        out = self.clf_conv(x1)
        return out

    def fit(self, data, labels, criterion):

        # forward pass through the model
        output = self.forward(data)

        # apply dense softmax
        output = F.softmax(output, dim=1)

        # interpolate output to match label data size
        output = F.interpolate(output, (labels.shape[-2], labels.shape[-1]), mode='bilinear', align_corners=True)

        # apply log
        output = torch.log(output)

        # reshape label tensor to 3D (N, H, W)
        labels = labels.view(-1, labels.shape[-2], labels.shape[-1])

        # compute loss from output and labels using criterion
        loss = criterion(output, labels)

        # do a backward pass to compute gradients
        self.enc_optimiser.zero_grad()
        self.dec_optimiser.zero_grad()
        loss.backward()

        # update model
        self.enc_optimiser.step()
        self.dec_optimiser.step()

        return loss

    def validate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # full confusion matrix
        full_cm = torch.zeros((dataset.num_classes, dataset.num_classes), dtype=torch.int64)
        if self.cuda_available:
            full_cm = full_cm.cuda()

        with torch.no_grad():
            for batch in dataloader:
                # extract data and labels from batch
                data = batch['data']
                labels = batch['label']
                if self.cuda_available:
                    data = data.cuda()
                    labels = labels.cuda()

                # forward pass through
                logits = self.forward(data)

                # interpolate logits back to original image size
                prediction = F.softmax(logits, dim=1)
                prediction = F.interpolate(prediction, (data.shape[-2], data.shape[-1]), mode='bilinear')
                prediction = torch.argmax(prediction, dim=1)
                prediction = torch.squeeze(prediction)

                # compute confusion matrix
                cm = compute_cm(labels, prediction, dataset.num_classes, self.cuda_available)
                full_cm += cm

            # compute mean IU from confusion matrix
            full_cm = full_cm.cpu().detach().numpy().astype(np.int64)
            iu = compute_iu(full_cm)
            mean_iu = np.mean(iu)

        return mean_iu

    def save(self, global_iteration, log_directory):
        os.makedirs(os.path.join(log_directory, 'snapshots'), exist_ok=True)
        model = {
            'model': self.state_dict(),
            'enc_optimiser': self.enc_optimiser.state_dict(),
            'dec_optimiser': self.dec_optimiser.state_dict(),
            'global_iteration': global_iteration
        }

        model_path = os.path.join(log_directory, 'snapshots', 'model-{:06d}.pth.tar'.format(global_iteration))
        print('Creating Snapshot: ' + model_path)
        torch.save(model, model_path)

    def load(self, log_directory=None, snapshot_num=None, with_optim=True):
        snapshot_dir = os.path.join(log_directory, 'snapshots')
        model_name = find_snapshot(snapshot_dir, snapshot_num)

        if model_name is None:
            print('Model not found: initialising using default PyTorch initialisation!')
            # uses pytorch default initialisation
            return 0
        # load model if snapshot was found
        else:
            full_model = torch.load(os.path.join(snapshot_dir, model_name))
            print('Loading model from: ' + os.path.join(snapshot_dir, model_name))
            self.load_state_dict(full_model['model'], strict=False)
            if with_optim:
                self.optimiser.load_state_dict(full_model['optimiser'])
                # move optimiser to cuda
                if self.cuda_available:
                    for state in self.optimiser.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
            curr_iteration = full_model['global_iteration']
            return curr_iteration


imagenet_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

pretrained_urls = {
    "refinenetlw50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
    "refinenetlw101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
    "refinenetlw152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
    "refinenetlw50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
    "refinenetlw101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
    "refinenetlw152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
}

# creates a ResNet-50 RefineNet (supports loading of pretrained ImageNet model)
def refinenet_lw50(num_classes, pretrained='imagenet', **kwargs):
    model = RefineNetLW(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

    if pretrained == 'nyu':
        key = 'refinenetlw50_nyu'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    elif pretrained == 'voc':
        key = 'refinenetlw50_voc'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    else:
        key = 'resnet50'
        url = imagenet_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    model.name = key
    return model

# creates a ResNet-101 RefineNet (supports loading of pretrained ImageNet model)
def refinenet_lw101(num_classes, pretrained='imagenet', **kwargs):
    model = RefineNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

    if pretrained == 'nyu':
        key = 'refinenetlw101_nyu'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    elif pretrained == 'voc':
        key = 'refinenetlw101_voc'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    else:
        key = 'resnet101'
        url = imagenet_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    model.name = key
    return model

# creates a ResNet-152 RefineNet (supports loading of pretrained ImageNet model)
def refinenet_lw152(num_classes, pretrained='imagenet', **kwargs):
    model = RefineNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)

    if pretrained == 'nyu':
        key = 'refinenetlw152_nyu'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    elif pretrained == 'voc':
        key = 'refinenetlw152_voc'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    else:
        key = 'resnet152'
        url = imagenet_urls[key]
        model.load_state_dict(download_model(key, url), strict=False)
    model.name = key
    return model


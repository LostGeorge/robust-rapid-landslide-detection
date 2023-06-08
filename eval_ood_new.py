from trainer_module.da_trainer import plDATrainerModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np

from models.da_models import instantiate_da_models
from models.discriminator import MLPDiscriminator
from data_module.multi_dm import MultiBeforeAfterCubeDataModule
from trainer_module.helpers import FocalTverskyLoss
import utils

from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dm_configs', nargs='+', type=str, default=['config/hokkaido.yaml', 'config/talakmau_70_30.yaml'])
    # parser.add_argument('--dm_configs', nargs='+', type=str, default=['config/hokkaido.yaml'])
    parser.add_argument('--encoder', type=str, default='resnet50')

    args = parser.parse_args()

    utils.seed_everything(0)
    
    # Load config files
    configs = [utils.parse_yaml(config_path) for config_path in args.dm_configs]

    dm = MultiBeforeAfterCubeDataModule(configs)

    n_channels = len(configs[0]['input_vars'])
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    encoder, models = instantiate_da_models(smp.UnetPlusPlus, args.encoder, num_channels=n_channels, classes=1, num_heads=2)
    encoder_out_size = utils.get_encoder_output_channels(args.encoder)
    discriminator = MLPDiscriminator(encoder_out_size, [encoder_out_size], len(models)).to(device)
    models = [model.to(device) for model in models]
    models[0].load_state_dict(torch.load('temp0.ckpt'))
    models[1].load_state_dict(torch.load('temp1.ckpt'))

    da_trainer_module = plDATrainerModule(encoder=encoder, models=models, discriminator=discriminator,
        model_losses={0: nn.CrossEntropyLoss()},
        model_evals={0: 0, 1: 0},
        lr=1e-3,
        device=device,
    )
    da_trainer_module = da_trainer_module.to(device)
    for i in range(2):
        da_trainer_module.models[i] = da_trainer_module.models[i].to(device)
        da_trainer_module.models[i].eval()
        da_trainer_module.models[i].decoder.train()
        da_trainer_module.models[i].segmentation_head.train()
    # checkpoint = torch.load('last.ckpt')
    # da_trainer_module.load_state_dict(checkpoint['state_dict'])
    # da_seg_model = da_trainer_module.models[0].to(device)
    # da_seg_model.eval()

    # trainer = pl.Trainer()
    # trainer.validate(da_trainer_module, dm)

    encoder, models = instantiate_da_models(smp.UnetPlusPlus, args.encoder, num_channels=n_channels, classes=1, num_heads=2)
    encoder_out_size = utils.get_encoder_output_channels(args.encoder)
    discriminator = MLPDiscriminator(encoder_out_size, [encoder_out_size], len(models)).to(device)
    models = [model.to(device) for model in models]
    models[0].load_state_dict(torch.load('temp0.ckpt'))
    models[1].load_state_dict(torch.load('temp1.ckpt'))

    sup_trainer_module = plDATrainerModule(encoder=encoder, models=models, discriminator=discriminator,
        model_losses={0: nn.CrossEntropyLoss()},
        model_evals={0: 0, 1: 0},
        lr=1e-3,
        device=device,
    )
    checkpoint = torch.load('lightning_logs/version_2/checkpoints/epoch=9-step=70.ckpt')
    sup_trainer_module.load_state_dict(checkpoint['state_dict'])
    sup_trainer_module = sup_trainer_module.to(device)
    for i in range(2):
        sup_trainer_module.models[i] = sup_trainer_module.models[i].to(device)
        sup_trainer_module.models[i].eval()
        sup_trainer_module.models[i].decoder.train()
        sup_trainer_module.models[i].segmentation_head.train()
    # sup_seg_model = sup_trainer_module.models[0].to(device)
    # sup_seg_model.eval()

    # trainer = pl.Trainer()
    # trainer.validate(sup_trainer_module, datamodule=dm)
    # trainer.validate(da_trainer_module, datamodule=dm)

    dm.prepare_data()
    dm.setup()
    val_loader_iter = iter(dm.val_dataloader()[1])
    # val_loader_iter = iter(dm.predict_dataloader()[0])
    x, target = next(val_loader_iter)
    x = x.to(device)
    target = target.long().to(device)

    gt = target.cpu().numpy()
    sup_pred_tens = sup_trainer_module.models[0](x)[:, 0]
    da_pred_tens = da_trainer_module.models[0](x)[:, 0]
    sup_pred = torch.round(torch.sigmoid(sup_pred_tens)).detach().cpu().numpy()
    da_pred = torch.round(torch.sigmoid(da_pred_tens)).detach().cpu().numpy()

    jaccard = JaccardIndex(task='binary').to(device)
    print(jaccard(sup_pred_tens, target).item())
    print(jaccard(da_pred_tens, target).item())
    
    gt_pos_ct = np.sum(gt, axis=(-1, -2))
    for i in range(len(gt_pos_ct)):
        sup_pred_iou = jaccard(sup_pred_tens[i], target[i])
        da_pred_iou = jaccard(da_pred_tens[i], target[i])
        print(i, gt_pos_ct[i], sup_pred_iou.item(), da_pred_iou.item())

    trainer = pl.Trainer()
    trainer.validate(da_trainer_module, dm)

    idx = 11
    gt_img = gt[idx]
    sup_pred_img = sup_pred[idx]
    da_pred_img = da_pred[idx]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    ax1.imshow(gt_img)
    ax1.set_title('Ground Truth', fontsize=12)
    ax1.set_axis_off()
    ax2.imshow(sup_pred_img)
    ax2.set_title('Without Domain Adaptation', fontsize=12)
    ax2.set_axis_off()
    ax3.imshow(da_pred_img)
    ax3.set_title('With Domain Adaptation', fontsize=12)
    ax3.set_axis_off()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0.2)
    fig.suptitle('Landslide Segmentation Output Maps', fontsize=15)
    # fig.tight_layout()

    plt.savefig('img.png')

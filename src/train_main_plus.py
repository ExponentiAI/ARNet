import os
import random
import argparse
import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_plus_utils import EncoderViT, get_acc, cross_loss, LoadMyDataset
from CNN_backbone import Backbone_VGG16, Backbone_Resnet50, Backbone_Inception


def train_model(args):
    if args.dataset == 'ClothesV1':
        image_path_train = './datasets/ClothesV1/trainB/'
        sketch_path_train = './datasets/ClothesV1/trainA/'
        save_path = './checkpoint/ClothesV1_plus/'

    elif args.dataset == 'ChairV2':
        image_path_train = './datasets/ChairV2/trainB/'
        sketch_path_train = './datasets/ChairV2/trainA/'
        save_path = './checkpoint/ChairV2_plus/'

    elif args.dataset == 'ShoeV2':
        image_path_train = './datasets/ShoeV2/trainB/'
        sketch_path_train = './datasets/ShoeV2/trainA/'
        save_path = './checkpoint/ShoeV2_plus/'

    else:
        raise ValueError('Dataset Name Error !')

    wandb.init(project='FGSBIR',
               config=args)

    start_epoch = 0
    end_epoch = args.num_epochs
    os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set = LoadMyDataset(img_folder_path=image_path_train,
                              skt_folder_path=sketch_path_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers, pin_memory=True)

    print('Dataset: {}  |  Batch size: {}\n'
          .format(args.dataset, args.batch_size))

    # img_model = EncoderViT(num_classes=args.num_classes, feature_dim=args.feature_dim,
    #                        encoder_backbone='vit_base_patch16_224')
    # skt_model = EncoderViT(num_classes=args.num_classes, feature_dim=args.feature_dim,
    #                        encoder_backbone='vit_base_patch16_224')

    img_model = EncoderViT(num_classes=args.num_classes, device=args.device)
    skt_model = EncoderViT(num_classes=args.num_classes, device=args.device)

    # img_model = Backbone_VGG16()
    # skt_model = Backbone_VGG16()

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        print('Loading Pretrained model successful !'
              'Epoch:[{}]  |  Loss:[{}]'.format(checkpoint['epoch'], checkpoint['loss']))
        print('Top1: {} %  |  Top5: {} %  |  Top10: {} %'.format(checkpoint['top1'], checkpoint['top5'],
                                                                 checkpoint['top10']))
        img_model.load_state_dict(checkpoint['img_model'])
        skt_model.load_state_dict(checkpoint['skt_model'])
        start_epoch = checkpoint['epoch']
        end_epoch = end_epoch + checkpoint['epoch']

    img_model.to(args.device)
    skt_model.to(args.device)

    scaler = GradScaler(enabled=args.fp16)
    optimizer = torch.optim.Adam([{"params": img_model.parameters()},
                                  {"params": skt_model.parameters()}],
                                 args.lr, weight_decay=args.weight_decay)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        wandb.log({'Progress': epoch}, step=epoch)
        epoch_train_contrastive_loss = 0
        epoch_cross_loss_anchor = 0
        epoch_cross_loss_self = 0
        epoch_cross_loss_triple = 0
        epoch_cross_loss_decor = 0

        img_model.train()
        skt_model.train()

        # 1.1 training for epochs
        for batch_idx, data in enumerate(tqdm(train_loader)):
            skt_anchor, skt_aug, img_anchor, img_aug = data
            skt_anchor, skt_aug = skt_anchor.to(args.device), skt_aug.to(args.device)
            img_anchor, img_aug = img_anchor.to(args.device), img_aug.to(args.device)

            optimizer.zero_grad()

            # 1.1.1 main contrastive loss
            skt_mlp_feat, skt_vit_feat, skt_decorrelation_loss = skt_model(skt_anchor, trainable=True)
            img_mlp_feat, img_vit_feat, img_decorrelation_loss = img_model(img_anchor, trainable=True)

            cross_loss_1 = cross_loss(skt_mlp_feat, img_mlp_feat, args)

            ###################### ViT #######################
            # 1.1.2 self loss
            skt_aug_feat = skt_model.embedding(skt_aug)[:, 0]
            img_aug_feat = img_model.embedding(img_aug)[:, 0]

            cross_loss_2 = cross_loss(skt_aug_feat, skt_vit_feat[:, 0], args)
            cross_loss_3 = cross_loss(img_aug_feat, img_vit_feat[:, 0], args)

            # # 1.1.3 contrastive loss
            cross_loss_4 = cross_loss(skt_vit_feat[:, 0], img_vit_feat[:, 0], args)
            ###################### ViT #######################

            # ####################### Swin #######################
            # # 1.1.2 self loss
            # skt_aug_feat = skt_model.embedding(skt_aug)
            # img_aug_feat = img_model.embedding(img_aug)
            #
            # cross_loss_2 = cross_loss(skt_aug_feat, skt_vit_feat, args)
            # cross_loss_3 = cross_loss(img_aug_feat, img_vit_feat, args)
            #
            # # # 1.1.3 contrastive loss
            # cross_loss_4 = cross_loss(skt_vit_feat, img_vit_feat, args)
            # ####################### Swin #######################

            cross_loss_5 = skt_decorrelation_loss + img_decorrelation_loss

            loss = cross_loss_1 + (cross_loss_2 + cross_loss_3) + cross_loss_4 + cross_loss_5

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_contrastive_loss = epoch_train_contrastive_loss + loss.item()
            epoch_cross_loss_anchor = epoch_cross_loss_anchor + cross_loss_1.item()
            epoch_cross_loss_self = epoch_cross_loss_self + (cross_loss_2 + cross_loss_3).item()
            epoch_cross_loss_triple = epoch_cross_loss_triple + cross_loss_4.item()
            epoch_cross_loss_decor = epoch_cross_loss_decor + cross_loss_5.item()

        print('Epoch Train: [{}] Contrastive Loss: {}'.format(epoch, epoch_train_contrastive_loss))
        wandb.log({'Contrastive Loss': epoch_train_contrastive_loss}, step=epoch)
        wandb.log({'Cross Loss Anchor': epoch_cross_loss_anchor}, step=epoch)
        wandb.log({'Self Loss': epoch_cross_loss_self}, step=epoch)
        wandb.log({'Triple Loss': epoch_cross_loss_triple}, step=epoch)
        wandb.log({'Decor Loss': epoch_cross_loss_decor}, step=epoch)

        img_model.eval()
        skt_model.eval()

        # 1.2 test for accuracy
        with torch.no_grad():
            print('Testing for dataset accuracy...')
            top1_accuracy, top5_accuracy, top10_accuracy = get_acc(skt_model, img_model, batch_size=128,
                                                                   dataset=args.dataset, mode='test',
                                                                   device=args.device)
            print('Top1: {:.3f} %  |  Top5: {:.3f} %  |  Top10: {:.3f} %'.format(top1_accuracy,
                                                                                 top5_accuracy,
                                                                                 top10_accuracy))
            wandb.log({'Top1 Acc': top1_accuracy}, step=epoch)
            wandb.log({'Top5 Acc': top5_accuracy}, step=epoch)
            wandb.log({'Top10 Acc': top10_accuracy}, step=epoch)

            top1_acc_train, top5_acc_train, top10_acc_train = get_acc(skt_model, img_model, batch_size=128,
                                                                      dataset=args.dataset, mode='train',
                                                                      device=args.device)
            print('Top1: {:.3f} %  |  Top5: {:.3f} %  |  Top10: {:.3f} %'.format(top1_acc_train,
                                                                                 top5_acc_train,
                                                                                 top10_acc_train))
            wandb.log({'Top1 Acc Train': top1_acc_train}, step=epoch)
            wandb.log({'Top5 Acc Train': top5_acc_train}, step=epoch)
            wandb.log({'Top10 Acc Train': top10_acc_train}, step=epoch)

        # 1.3 save checkpoints
        if top1_accuracy > args.best_top1_acc:
            args.best_top1_acc = top1_accuracy
            args.best_top5_acc = top5_accuracy
            args.best_top10_acc = top10_accuracy
            save_state = {'img_model': img_model.state_dict(),
                          'skt_model': skt_model.state_dict(),
                          'epoch': epoch,
                          'loss': round(epoch_train_contrastive_loss, 5),
                          'top1': top1_accuracy,
                          'top5': top5_accuracy,
                          'top10': top10_accuracy}
            print('Updating Model checkpoint [Best Acc]...')
            torch.save(save_state, os.path.join(save_path, 'model_Best.pth'))

        if top1_accuracy == args.best_top1_acc:
            if top5_accuracy > args.best_top5_acc:
                args.best_top1_acc = top1_accuracy
                args.best_top5_acc = top5_accuracy
                save_state = {'img_model': img_model.state_dict(),
                              'skt_model': skt_model.state_dict(),
                              'epoch': epoch,
                              'loss': round(epoch_train_contrastive_loss, 5),
                              'top1': top1_accuracy,
                              'top5': top5_accuracy,
                              'top10': top10_accuracy}
                print('Updating Network checkpoint...')
                torch.save(save_state, os.path.join(save_path, 'model_' + str(epoch) + '.pth'))
            elif top10_accuracy > args.best_top10_acc:
                args.best_top1_acc = top1_accuracy
                args.best_top10_acc = top10_accuracy
                save_state = {'img_model': img_model.state_dict(),
                              'skt_model': skt_model.state_dict(),
                              'epoch': epoch,
                              'loss': round(epoch_train_contrastive_loss, 5),
                              'top1': top1_accuracy,
                              'top5': top5_accuracy,
                              'top10': top10_accuracy}
                print('Updating Network checkpoint...')
                torch.save(save_state, os.path.join(save_path, 'model_' + str(epoch) + '.pth'))

    print('Best Acc:\nTop1: {:.3f} %  |  Top5: {:.3f} %  |  Top10: {:.3f} %'.format(args.best_top1_acc,
                                                                                    args.best_top5_acc,
                                                                                    args.best_top10_acc))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for FGSBIR Network')
    parser.add_argument('--dataset', default='ChairV2', help='ClothesV1, ChairV2, ShoeV2')
    parser.add_argument('--num_classes', type=int, default=512, help='num classes')
    parser.add_argument('--feature_dim', type=int, default=768, help='ouput feature dim')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='data loader batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--num_epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--save_iter', type=int, default=100, help='the training iter to save model')
    parser.add_argument('--lr', type=float, default=6e-6, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='learning rate weight decay')
    parser.add_argument('--best_top1_acc', type=float, default=0.0, help='the best training Top1 acc')
    parser.add_argument('--best_top5_acc', type=float, default=0.0, help='the best training Top5 acc')
    parser.add_argument('--best_top10_acc', type=float, default=0.0, help='the best training Top10 acc')
    parser.add_argument('--temperature', type=float, default=0.07, help='softmax temperature')
    parser.add_argument('--fp16', type=bool, default=True, help='if use the fp16 precision')
    parser.add_argument('--shuffle', type=bool, default=True, help='if shuffle datasets')
    parser.add_argument('--device', type=str, default='cuda:0', help='training device')
    parser.add_argument('--n-views', type=int, default=2, help='Number of views for contrastive learning.')
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained model checkpoint path')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)

    train_model(args)

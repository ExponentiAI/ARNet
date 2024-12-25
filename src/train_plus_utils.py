"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import os
import glob
import random
import numpy as np
from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer
import torch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import transforms


# # Swin-ViT
# class EncoderViT(nn.Module):
#     def __init__(self, num_classes=512, embed_dim=1024, encoder_backbone='swin_base_patch4_window7_224'):
#         super().__init__()
#         self.encoder: SwinTransformer = timm.create_model(encoder_backbone, pretrained=True)
#         self.mlp_head = nn.Sequential(
#             nn.Linear(embed_dim, num_classes)
#         )
#
#     def embedding(self, x):
#         x = self.encoder.patch_embed(x)  # (B, 3136, 128)
#         if self.encoder.absolute_pos_embed is not None:
#             x = x + self.encoder.absolute_pos_embed
#         x = self.encoder.pos_drop(x)
#         x = self.encoder.layers(x)  # (B, 49, 1024)  B L C
#         x = self.encoder.norm(x)
#         x = self.encoder.avgpool(x.transpose(1, 2))  # (B, 1024, 1)  B C 1
#         x = torch.flatten(x, 1)  # (B, 1024)
#
#         return x
#
#     def forward(self, x):  # (B, 3, 224, 224)
#         x = self.embedding(x)
#         cls = self.mlp_head(x)  # (B, cls)
#
#         return cls, x


# # # ViT
# class EncoderViT(nn.Module):
#     def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224'):
#         super().__init__()
#         self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)
#         self.mlp_head = nn.Sequential(
#             nn.Linear(feature_dim, num_classes)
#         )
#
#         for m in self.mlp_head.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#         self.special_feature = nn.Parameter(torch.randn(196, 1, 768))
#         self.filter = nn.Linear(768, 10)
#
#     def embedding(self, image):
#         x = self.encoder.patch_embed(image)
#         cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
#         if self.encoder.dist_token is None:
#             x = torch.cat((cls_token, x), dim=1)
#         else:
#             x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#         x = self.encoder.pos_drop(x + self.encoder.pos_embed)
#         x = self.encoder.blocks(x)
#         x = self.encoder.norm(x)
#         return x
#
#     def forward(self, image):
#         vit_feat = self.embedding(image)
#         mlp_feat = self.mlp_head(vit_feat[:, 0])
#         return mlp_feat, vit_feat


# # ViT
class EncoderViT(nn.Module):
    def __init__(self, num_classes=256, feature_dim=768, encoder_backbone='vit_base_patch16_224', device='cuda'):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=True)
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.recycle_model = MultiScaleTransformer(scales=[1, 2, 7], device=device)

    def embedding(self, image):
        x = self.encoder.patch_embed(image)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward(self, image, trainable=False):
        vit_feat = self.embedding(image)

        if trainable:
            recycle_feat, decorrelation_loss = self.recycle_model(vit_feat, trainable)
            mlp_feat = self.mlp_head(recycle_feat)

            return mlp_feat, vit_feat, decorrelation_loss

        else:
            recycle_feat = self.recycle_model(vit_feat, trainable)
            mlp_feat = self.mlp_head(recycle_feat)

            return mlp_feat, vit_feat


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiScaleTransformer(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, scales=[1, 2, 7], device='cuda'):
        super(MultiScaleTransformer, self).__init__()
        self.transformer_models = [TransformerEncoder(input_dim=input_dim, num_heads=num_heads,
                                                      num_layers=2).to(device)
                                   for _ in scales]
        self.scales = scales
        self.fc = nn.Linear(196 + 49 + 4, 1)
        # self.weight = nn.Parameter(torch.tensor(0.1))
        self.weight = 0.1

    def forward(self, vit_features, trainable=False):
        cls_token = vit_features[:, 0, :]  # (B, 768)
        patch_tokens = vit_features[:, 1:, :]  # (B, 196, 768)

        multi_scale_features = []
        for scale, transformer in zip(self.scales, self.transformer_models):
            if scale != 1:
                B, N, C = patch_tokens.size()
                new_size = int(N / (scale * scale))
                patch_tokens_scaled = patch_tokens.view(B, new_size, scale, scale, C)
                patch_tokens_scaled = patch_tokens_scaled.mean(dim=(2, 3))
            else:
                patch_tokens_scaled = patch_tokens
            enhanced_patch_tokens = transformer(patch_tokens_scaled)  # (B, new_size, 768)
            multi_scale_features.append(enhanced_patch_tokens)

        combined_features = torch.cat(multi_scale_features, dim=1)  # (B, 196 + 49 + 4, 768)

        fc_features = self.fc(combined_features.contiguous().permute(0, 2, 1)).squeeze(2)  # (B, 768)

        final_features = (1.0 - self.weight) * cls_token + self.weight * fc_features

        if trainable:
            pooled_features = nn.functional.avg_pool1d(combined_features, kernel_size=48)

            similarity_matrix = torch.matmul(pooled_features, pooled_features.transpose(-1, -2))  # (B, 249, 249)

            identity_matrix = self.contrast_matrix(similarity_matrix.size(-1)).to(similarity_matrix.device)
            identity_matrix = identity_matrix.unsqueeze(0).expand(similarity_matrix.size(0), -1, -1)
            decorrelation_loss = F.mse_loss(similarity_matrix, identity_matrix)

            return final_features, decorrelation_loss

        else:
            return final_features

    def contrast_matrix(self, matrix_len):
        identity_matrix = torch.eye(matrix_len)

        matrix_10 = torch.zeros((49, 196))
        for i in range(49):
            matrix_10[i, i*4:(i+1)*4] = 0.5

        matrix_20 = torch.zeros((4, 245))
        for i in range(4):
            matrix_20[i, i * 49:(i + 1) * 49] = 0.5
        for i in range(4):
            if i != 3:
                matrix_20[i, 196 + i*12:196 + (i + 1) * 13] = 0.5
            else:
                matrix_20[i, 196 + i * 12:] = 0.5

        matrix_01 = torch.zeros((196, 49))
        for i in range(49):
            matrix_01[i * 4:(i + 1) * 4, i] = 0.5

        matrix_02 = torch.zeros((245, 4))
        for i in range(4):
            matrix_02[i * 49:(i + 1) * 49, i] = 0.5
        for i in range(4):
            if i != 3:
                matrix_20[196 + i * 12:196 + (i + 1) * 13, i] = 0.5
            else:
                matrix_20[196 + i * 12:, i] = 0.5

        identity_matrix[196:245, 0:196] = matrix_10
        identity_matrix[245:249, 0:245] = matrix_20
        identity_matrix[0:196, 196:245] = matrix_01
        identity_matrix[0:245, 245:249] = matrix_02

        # identity_matrix = identity_matrix.numpy()

        return identity_matrix


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total params: ', total_num, '\nTrainable params: ', trainable_num)


class LoadMyDataset(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, im_size=224):
        skt_list = []
        img_list = []
        # skt_pos_list = []
        img_name_list = os.listdir(img_folder_path)

        for pos_img_name in img_name_list:
            # 1.1 anchor_skt_name, pos_skt_name
            pos_skt_list_path = glob.glob(skt_folder_path + pos_img_name.split('.')[0] + '_?.png')
            pos_skt_list_name = [file_name.split('/')[-1].split('.')[0] + '.png'
                                 for file_name in pos_skt_list_path]

            # 1.3 append lists
            if len(pos_skt_list_name) == 0:
                print(pos_img_name)

            for len_list in range(len(pos_skt_list_name)):
                skt_list.append(pos_skt_list_name[len_list])
                img_list.append(pos_img_name)

            # if len(pos_skt_list_name) == 0:
            #     continue
            #
            # elif len(pos_skt_list_name) == 1:
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[0])
            #
            # else:
            #     random_idx = random.randint(1, len(pos_skt_list_name) - 1)
            #     skt_list.append(pos_skt_list_name[0])
            #     img_list.append(pos_img_name)
            #     skt_pos_list.append(pos_skt_list_name[random_idx])

        self.img_folder_path = img_folder_path
        self.skt_folder_path = skt_folder_path

        self.skt_list = skt_list
        self.img_list = img_list
        # self.skt_pos_list = skt_pos_list

        self.transform_anchor = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        self.transform_aug = transforms.Compose([
            transforms.Resize((int(im_size * 1.2), int(im_size * 1.2))),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.CenterCrop((int(im_size), int(im_size))),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor()
        ])

        # self.transform_aug = transforms.Compose([
        #     transforms.RandomResizedCrop(size=(224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

    def __getitem__(self, item):
        # 1.1 anchor skt
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        skt_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))
        skt_aug = self.transform_aug(Image.fromarray(np.array(Image.open(skt_path).convert('RGB'))))

        # 1.2 anchor img
        pos_path = os.path.join(self.img_folder_path, self.img_list[item])
        img_anchor = self.transform_anchor(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))
        img_aug = self.transform_aug(Image.fromarray(np.array(Image.open(pos_path).convert('RGB'))))

        sample = skt_anchor, skt_aug, img_anchor, img_aug

        return sample

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetSkt(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        skt_list = []
        label_list = []
        img_name_list = os.listdir(img_folder_path)

        for img_name in img_name_list:
            skt_list_path = glob.glob(skt_folder_path + img_name.split('.')[0] + '_?.png')
            for skt_name in skt_list_path:
                skt_name = skt_name.split('/')[-1].split('.')[0] + '.png'
                img_item = img_name_list.index(img_name)
                skt_list.append(skt_name)
                label_list.append(img_item)

        self.skt_folder_path = skt_folder_path
        self.transform = transform
        self.skt_list = skt_list
        self.label_list = label_list

    def __getitem__(self, item):
        skt_path = os.path.join(self.skt_folder_path, self.skt_list[item])
        sample_skt = self.transform((Image.fromarray(np.array(Image.open(skt_path).convert('RGB')))))
        image_idx = self.label_list[item]

        return sample_skt, image_idx

    def __len__(self):
        return len(self.skt_list)


class LoadDatasetImg(Dataset):
    def __init__(self, img_folder_path, skt_folder_path, transform):
        self.transform = transform
        self.img_folder_path = img_folder_path
        self.img_list = os.listdir(img_folder_path)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_folder_path, self.img_list[item])
        sample_img = self.transform((Image.fromarray(np.array(Image.open(img_path).convert('RGB')))))

        return sample_img

    def __len__(self):
        return len(self.img_list)


def get_acc(skt_model, img_model, batch_size=128, dataset='ClothesV1', mode='test', device='cuda'):
    print('Evaluating Network dataset [{}_{}] ...'.format(dataset, mode))

    data_set_skt = LoadDatasetSkt(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_set_img = LoadDatasetImg(img_folder_path='./datasets/{}/{}B/'.format(dataset, mode),
                                  skt_folder_path='./datasets/{}/{}A/'.format(dataset, mode),
                                  transform=Compose([Resize(224), ToTensor()]))

    data_loader_skt = DataLoader(data_set_skt, batch_size=batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
    data_loader_img = DataLoader(data_set_img, batch_size=batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)

    skt_model = skt_model.to(device)
    img_model = img_model.to(device)
    skt_model.eval()
    img_model.eval()

    top1_count = 0
    top5_count = 0
    top10_count = 0

    with torch.no_grad():
        Image_Feature = torch.FloatTensor().to(device)
        for imgs in tqdm(data_loader_img):
            img = imgs.to(device)
            img_feats, _ = img_model(img)
            img_feats = F.normalize(img_feats, dim=1)
            Image_Feature = torch.cat((Image_Feature, img_feats.detach()))

        for idx, skts in enumerate(tqdm(data_loader_skt)):
            skt, skt_idx = skts
            skt, skt_idx = skt.to(device), skt_idx.to(device)
            skt_feats, _ = skt_model(skt)
            skt_feats = F.normalize(skt_feats, dim=1)

            similarity_matrix = torch.argsort(torch.matmul(skt_feats, Image_Feature.T), dim=1, descending=True)

            top1_count += (similarity_matrix[:, 0] == skt_idx).sum()
            top5_count += (similarity_matrix[:, :5] == torch.unsqueeze(skt_idx, dim=1)).sum()
            top10_count += (similarity_matrix[:, :10] == torch.unsqueeze(skt_idx, dim=1)).sum()

        top1_accuracy = round(top1_count.item() / len(data_set_skt) * 100, 3)
        top5_accuracy = round(top5_count.item() / len(data_set_skt) * 100, 3)
        top10_accuracy = round(top10_count.item() / len(data_set_skt) * 100, 3)

    return top1_accuracy, top5_accuracy, top10_accuracy


#  InfoNCE Loss
def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    # normalize
    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    encoder = EncoderViT(num_classes=256, feature_dim=768,
                         encoder_backbone='vit_base_patch16_224', device=device).to(device)
    get_parameter_number(encoder)
    # Total params: 87,423,541
    # Trainable params: 87,423,541

    img = torch.randn((2, 3, 224, 224)).to(device)
    out1, out2 = encoder(img, trainable=True)

    print('Done !')

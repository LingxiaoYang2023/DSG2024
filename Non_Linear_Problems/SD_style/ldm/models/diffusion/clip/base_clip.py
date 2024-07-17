import torch
import torch.nn as nn
from .clip import clip
# from clip import clip
import torchvision
from PIL import Image

model_name = "ViT-B/16"
# model_name = "ViT-B/32"


def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPEncoder(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad = True
        # self.transform = torchvision.transforms.Normalize(
        #     (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
        #     (0.26862954*2, 0.26130258*2, 0.27577711*2)
        # )
        self.transform = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                          (0.26862954, 0.26130258, 0.27577711))

        # if need_ref:
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
            
            # img = Image.open(ref_path).convert('RGB')
            # image = img.resize((224, 224), Image.Resampling.BILINEAR)
            # img = self.to_tensor(image)
            # img = torch.unsqueeze(img, 0)
            # img = img.cuda()
            # self.ref = img
    def preprocess_from_path(self,img_path):
        img = Image.open(img_path).convert('RGB')
        image = img.resize((224, 224), Image.Resampling.BILINEAR)
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        return img

    def preprocess_from_tensor(self,img):
        img = (img + 1.0) / 2.0
        image = torch.nn.functional.interpolate(img, size=(224, 224), mode='bicubic')
        img = self.transform(image)
        return img

    def get_gram_matrix_residual(self, im1, im2):
        # im1 = torch.nn.functional.interpolate(im1, size=(224, 224), mode='bicubic')
        # im1 = self.preprocess(im1)
        # im1 = self.preprocess_from_tensor(im1)
        # im2 = self.preprocess_from_path(im2_path)

        f1, feats1 = self.clip_model.encode_image_with_features(im1)
        f2, feats2 = self.clip_model.encode_image_with_features(im2)
        
        feat1 = feats1[2][1:, 0, :]
        feat2 = feats2[2][1:, 0, :]
        gram1 = torch.mm(feat1.t(), feat1)
        gram2 = torch.mm(feat2.t(), feat2)
        return gram1 - gram2

    def get_residual(self, image, text):
        text = clip.tokenize(text).to(self.device)
        # image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        # image = self.transform(image)
        image_feature, _ = self.clip_model.encode_image_with_features(image)
        text_feature = self.clip_model.encode_text(text)
        text_feature = text_feature.repeat(image.shape[0], 1)
        return text_feature - image_feature



if __name__ == "__main__":
    m = CLIPEncoder().cuda()
    im1 = torch.randn((1, 3, 224, 224)).cuda()
    im2 = torch.randn((1, 3, 224, 224)).cuda()
    m.get_gram_matrix_residual(im1, im2)

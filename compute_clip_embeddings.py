from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
import numpy as np
from datasets import load_dataset
model = AutoModel.from_pretrained(
    "openai/clip-vit-base-patch32", cache_dir=".cache"
).cuda()
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=".cache")
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir=".cache",
)

class Transform_CLIP(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            # transforms.ToTensor(),
            transforms.Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean, std),
        )
        

    def forward(self, x):
        with torch.no_grad():
            x = self.preprocess(x)
        return x
image_transformations_clip = Transform_CLIP(
    model.config.vision_config.image_size, feature_extractor.image_mean, feature_extractor.image_std
)

print("loading dataset")
data = pd.read_csv("caption_stage1_train_drop.csv", delimiter=',', encoding="utf-8")
print("dataset loaded")

nb_img = len(data.index)
caption_embeds = torch.zeros((nb_img, 512), dtype=torch.float32, device="cuda")
img_embeds = torch.zeros((nb_img, 512), dtype=torch.float32, device="cuda")

convert_tensor = transforms.ToTensor()
u = 0 
batch_size = 8192
pbar = tqdm(total = nb_img)
while(u + batch_size <= nb_img): 

    data_rows = data.iloc[u:u+batch_size]
    captions = [caption.split("&&")[0].strip() for caption in data_rows["caption"]]
    text_inputs = tokenizer(captions, padding="longest", max_length=256, truncation=True, return_tensors="pt")
    images = [convert_tensor(Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')) for img_data in data_rows["img"]]
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=torch.stack([image_transformations_clip(image) for image in images]).to(model.device),
            return_dict=model.config.return_dict
        )
        image_embeds = vision_outputs[1]
        image_embeds = model.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"].cuda(),
            attention_mask=text_inputs["attention_mask"].cuda(),
        )

        text_embeds = text_outputs[1]
        text_embeds = model.text_projection(text_embeds)

        # normalized features
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    caption_embeds[u:u+batch_size] = text_embeds
    img_embeds[u:u+batch_size] = image_embeds

    u = u + batch_size
    pbar.update(batch_size)
if(u!=nb_img):
    data_rows = data.iloc[u:]
    captions = [caption.split("&&")[0].strip() for caption in data_rows["caption"]]
    text_inputs = tokenizer(captions, padding="longest", max_length=256, truncation=True, return_tensors="pt")
    images = [convert_tensor(Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')) for img_data in data_rows["img"]]

    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=torch.stack([image_transformations_clip(image) for image in images]).to(model.device),
            return_dict=model.config.return_dict
        )
        image_embeds = vision_outputs[1]
        image_embeds = model.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        text_outputs = model.text_model(
            input_ids=text_inputs["input_ids"].cuda(),
            attention_mask=text_inputs["attention_mask"].cuda(),
        )

        text_embeds = text_outputs[1]
        text_embeds = model.text_projection(text_embeds)

        # normalized features
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    caption_embeds[u:] = text_embeds
    img_embeds[u:] = image_embeds


    pbar.update(len(data_rows))
caption_embeds =caption_embeds.cpu().numpy()
img_embeds = img_embeds.cpu().numpy()
with open("embeddings/caption_embeddings_train.npy", 'wb') as f:
    np.save(f, caption_embeds, allow_pickle=False)

with open("embeddings/image_embeddings_train.npy", 'wb') as f:
    np.save(f, img_embeds, allow_pickle=False)


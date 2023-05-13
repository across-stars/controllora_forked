import os
import cv2
import glob
import math
import torch
import datasets
import scipy.io
import jsonlines
import numpy as np

from tqdm import tqdm
from PIL import Image
from clip_interrogator import Config, Interrogator
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


MAIN_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

img_dir = f'{MAIN_DIR}/data/COCO_train2017/images'

jlname = f"{MAIN_DIR}/data/COCO_train2017/annotations_CLIP.jsonl"
def make_prompt():
    imgs = sorted(glob.glob(f'{img_dir}/**/*.jpg', recursive=True))[:20000]
    if not os.path.exists(jlname + '.done'):
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        with jsonlines.open(jlname, 'w', flush=True) as w:
            for img_path in tqdm(imgs):
                with torch.no_grad():
                    image = Image.open(img_path).convert('RGB')
                    prompt = ci.interrogate_fast(image)
                    w.write({
                        "image": img_path, 
                        "text": prompt}
                    )
        with open(jlname + '.done', 'w', encoding='utf-8') as f:
            f.write('Done.')

def create_dataset_from_json():
    dataset = Dataset.from_json(jlname)
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    dataset = DatasetDict(train=dataset)
    dataset.save_to_disk(os.path.join(MAIN_DIR, 'data/COCO'))


make_prompt()
create_dataset_from_json()


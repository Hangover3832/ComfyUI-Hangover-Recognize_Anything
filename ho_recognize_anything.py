"""
@author: AlexL
@title: ComfyUI-Hangover-Recognize_Anything
@nickname: Hangover-Recognize_Anything
@description: An implementation of the Recognize Anything Model (RAM++) for ComfyUI
"""

# https://huggingface.co/xinyu1205/recognize-anything-plus-model
# https://github.com/xinyu1205/recognize-anything

# by https://github.com/Hangover3832


from PIL import Image
import torch
import gc
import numpy as np
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from folder_paths import models_dir

class RecognizeAnything:
    MODEL_NAMES = ["ram_plus_swin_large_14m.pth",] # other/newer models can be added here
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]
    IMAGE_SIZES = [384, 448, 512, 576]

    def __init__(self):
        self.model = None
        self.modelname = ""
        self.image_size = 384

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (s.MODEL_NAMES, {"default": s.MODEL_NAMES[0]},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
                "中文": ("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def interrogate(self, image:torch.Tensor, model:str, device:str, 中文:bool):
        dev = "cuda" if device.lower() == "gpu" else "cpu"
        model_path = models_dir + '/' + "ram/" + model

        if (self.model == None) or (self.modelname != model) or (device != self.device):
            del self.model
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            print(f"RAM++: loading model {model_path}, please stand by....")
            self.transform = get_transform(image_size=RecognizeAnything.IMAGE_SIZES[0])
            self.model = ram_plus(pretrained=model_path, image_size=self.image_size, vit='swin_l')
            self.model.eval()
            self.model = self.model.to(dev)

            self.modelname = model
            self.device = device

        tags = ""
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            image = self.transform(img).unsqueeze(0).to(dev)
            res = inference(image, self.model)
            tag = res[1] if 中文 else res[0]
            tags += tag.replace(' ', '').replace('|', ',') + '\n'
        
        return(tags,)

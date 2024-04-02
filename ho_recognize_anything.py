"""
@author: AlexL
@title: ComfyUI-Hangover-Recognize_Anything
@nickname: Hangover-Recognize_Anything
@description: An implementation of the Recognize Anything Model (RAM++) for ComfyUI. The counterpart of Segment Anything Model (SAM).
"""

# https://huggingface.co/xinyu1205/recognize-anything-plus-model
# https://github.com/xinyu1205/recognize-anything

# by https://github.com/Hangover3832


from PIL import Image
import torch
import gc
import numpy as np
from ram.models import ram
from ram.models import ram_plus
from ram.models import tag2text
from ram import inference_ram # as inference
from ram import inference_tag2text # as inference
from ram import get_transform
from folder_paths import models_dir
from functools import partial

class RecognizeAnything:
    MODEL_NAMES = ["ram_swin_large_14m.pth", "ram_plus_swin_large_14m.pth", "tag2text_swin_14m.pth"] # other/newer models can be added here
    DEVICES = ["cpu", "gpu"] if torch.cuda.is_available() else  ["cpu"]
    IMAGE_SIZES = [384, 448]

    def __init__(self):
        self.model = None
        self.modelname = ""
        self.image_size = 384
        self.inference = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (s.MODEL_NAMES, {"default": s.MODEL_NAMES[1]},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},),
                #"中文": ("BOOLEAN", {"default": False},),
                "spec_tag2text": ("STRING", {"default": ""},),
                # "tag2text_threshold": ("INT", {"min": 0, "max": 100, "default": 68},),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("tags", "spec_tags", "caption")
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "Hangover"

    def interrogate(self, image:torch.Tensor, model:str, device:str, spec_tag2text:str):
        dev = torch.device("cuda" if device.lower() == "gpu" else "cpu")
        model_path = f"{models_dir}/rams/{model}"
        self.transform = get_transform(image_size=RecognizeAnything.IMAGE_SIZES[0])

        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

        if (self.model == None) or (self.modelname != model) or (device != self.device):
            del self.model
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            print(f"RAM++: loading model {model_path}, please stand by....")

            if model == RecognizeAnything.MODEL_NAMES[0]:
                self.model = ram(pretrained=model_path, image_size=self.image_size, vit='swin_l')
                self.inference = inference_ram
                spec_tag2text = None
            elif model == RecognizeAnything.MODEL_NAMES[1]:
                self.model = ram_plus(pretrained=model_path, image_size=self.image_size, vit='swin_l')
                self.inference = inference_ram
                spec_tag2text = None
            elif model == RecognizeAnything.MODEL_NAMES[2]:
                self.model = tag2text(pretrained=model_path, image_size=self.image_size, vit='swin_b', delete_tag_index=delete_tag_index)
                # self.model.threshold = tag2text_threshold  # threshold for tagging
                self.inference = inference_tag2text
            else:
                raise ValueError('No valid model was selected', model)

            self.model.eval()
            self.model = self.model.to(dev)

            self.modelname = model
            self.device = device

        tag0 = tag1 = tag2 = ''
        
        for im in image:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            image = self.transform(img).unsqueeze(0).to(dev)
            if  model == RecognizeAnything.MODEL_NAMES[2]:
                res = self.inference(image, self.model, input_tag=spec_tag2text)
            else:
                res = self.inference(image, self.model)
            tag0 += res[0].replace(' ', '').replace('|', ',') + '\n'
            tag1 += res[1].replace(' ', '').replace('|', ',') + '\n' if (len(res) > 1) and res[1] else '\n'
            tag2 += res[2] + '\n' if (len(res) > 2) and res[2] else '\n'
        
        return(tag0, tag1, tag2,)

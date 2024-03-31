from .ho_recognize_anything import RecognizeAnything

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique

NODE_CLASS_MAPPINGS = {
    "Recognize Anything Model (RAM++)": RecognizeAnything,
}

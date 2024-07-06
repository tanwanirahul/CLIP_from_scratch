
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

from models import CLIP


def load_images():
    '''
        loads images to be used as input for the CLIP model.
    '''
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image1 = Image.open(requests.get(url, stream=True).raw)

    url = "http://images.cocodataset.org/test-stuff2017/000000000019.jpg"
    image2 = Image.open(requests.get(url, stream=True).raw)

    return [image1, image2]


if __name__ == "__main__":
    
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)

    print(f"Loading OpenAI's CLIP from HF.")
    hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print(f"Creating our CLIP model from the pre-trained weights of HF model.")
    model = CLIP.from_pretrained(hf_model)


    print(f"Preparing data for evaluation.")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    images = load_images()
    inputs = processor(text=["a photo of a cat", "an another photo of a cow"], images=images, return_tensors="pt", padding=True)

    print(f"Running evaluation on HF transformer model.")
    hf_outputs = hf_model(**inputs,return_loss=True)
    print(F"Running evaluation on our CLIP implementation.")
    outputs = model(**inputs, return_loss=True)

    hf_logits_per_image = hf_outputs.logits_per_image  # this is the image-text similarity score
    hf_probs = hf_logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    hf_loss = hf_outputs.loss
    
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label pr
    loss = outputs.loss
    print(f"\nHF transformer model output: ")
    print(f"loss -> {hf_loss};\nprobs -> {hf_probs}")
    print(f"\nOur CLIP model output: ")
    print(f"loss -> {loss};\nprobs -> {probs}")
    
    is_equal = torch.allclose(loss, hf_loss, atol=1e-5) and torch.allclose(probs, hf_probs, atol=1e-5)
    print(f"\nDo outputs match?: {is_equal}\n")
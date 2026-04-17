image_path = './demo/demo.jpg'
sentence = 'the most handsome guy'
weights = './checkpoints/refcoco.pth'
device = 'cuda:0'

# pre-process the input image
from PIL import Image
import torchvision.transforms as T
import numpy as np
img = Image.open(image_path).convert("RGB")
img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
original_w, original_h = img.size  # PIL .size returns width first and height second

image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
img = img.to(device)  # for inference (input)

# pre-process the raw sentence
import torch

from lib import segmentation
from text_encoder import (build_text_encoder, build_text_tokenizer, encode_text,
                          get_checkpoint_text_encoder_state, prepare_text_encoder_args,
                          tokenize_text)
from utils import load_state_dict_with_fallback


class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0
    align_module = 'pwam'
    gate_module = 'lg'
    hapwam_hidden_dim = 256
    hapwam_fusion_hidden_dim = 256
    hapwam_dropout = 0.1
    hlg_hidden_channels = None
    hlg_stages = [3, 4]
    text_encoder_name = 'microsoft/deberta-v3-base'
    text_tokenizer_name = ''
    max_text_tokens = 64


prepare_text_encoder_args(args)
tokenizer = build_text_tokenizer(args)
padded_sent_toks, attention_mask = tokenize_text(tokenizer, sentence, args.max_text_tokens)
padded_sent_toks = padded_sent_toks.to(device)
attention_mask = attention_mask.to(device)

single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
single_model.to(device)
single_text_encoder = build_text_encoder(args)

checkpoint = torch.load(weights, map_location='cpu')
text_encoder_state, text_encoder_key = get_checkpoint_text_encoder_state(checkpoint, args)
if text_encoder_state is not None:
    load_state_dict_with_fallback(single_text_encoder,
                                  text_encoder_state,
                                  strict=True,
                                  description=text_encoder_key)
load_state_dict_with_fallback(single_model, checkpoint['model'], strict=True, description='model')
model = single_model.to(device)
text_encoder = single_text_encoder.to(device)


# inference
import torch.nn.functional as F
last_hidden_states = encode_text(text_encoder, padded_sent_toks, attention_mask)
embedding = last_hidden_states.permute(0, 2, 1)
output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
output = output.squeeze()  # (orig_h, orig_w)
output = output.cpu().data.numpy()  # (orig_h, orig_w)


# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
# Overlay the mask on the image
visualization = overlay_davis(img_ndarray, output)  # red
visualization = Image.fromarray(visualization)
# show the visualization
#visualization.show()
# Save the visualization
visualization.save('./demo/demo_result.jpg')

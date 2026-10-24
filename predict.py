import argparse
import json
import torch
from PIL import Image, ImageOps
from config import Config
from models.factory import build_model
from data_aug import transform


class Predictor:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            raise ValueError('Legacy checkpoint: architecture and preprocessing metadata must be migrated first')
        self.cfg = Config(**checkpoint['config']).validate()
        self.classes = checkpoint['classes']
        if len(self.classes) != self.cfg.num_classes: raise ValueError('Invalid checkpoint class labels')
        self.class_to_idx = checkpoint['class_to_idx']
        self.model = build_model(self.cfg, initialize_pretrained=False)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.spec = checkpoint['preprocessing']
        self.transform = transform(self.spec)

    @torch.inference_mode()
    def predict(self, image):
        if not isinstance(image, Image.Image):
            with Image.open(image) as opened: image = ImageOps.exif_transpose(opened).convert('RGB')
        else: image = ImageOps.exif_transpose(image).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if not torch.isfinite(logits).all(): raise ValueError('Non-finite model output')
        scores = logits.softmax(1)[0].cpu().tolist()
        index = max(range(len(scores)), key=scores.__getitem__)
        return {'class': self.classes[index], 'confidence': scores[index] * 100,
                'all_classes': self.classes, 'all_confidences': [v * 100 for v in scores]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True); parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cpu'); args = parser.parse_args()
    print(json.dumps(Predictor(args.checkpoint, args.device).predict(args.image), indent=2))

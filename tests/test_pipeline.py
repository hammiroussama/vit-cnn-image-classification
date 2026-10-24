"""Offline regression checks. Synthetic images test correctness, not accuracy."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from data import stratified_split, training_loaders
from data_aug import transform, preprocessing, mixing
from models.factory import build_model, parameter_groups
from metrics import classification_metrics
from train import train_epoch, run
from predict import Predictor
from app import create_app
from werkzeug.test import Client
from werkzeug.wrappers import Response

torch.set_num_threads(2)


def tiny_config(**kwargs):
    values = dict(model_name='vit', img_size=32, patch_size=4, embed_dim=32,
                  depth=1, num_heads=4, batch_size=2, epochs=1, num_workers=0,
                  mixup_alpha=0, cutmix_alpha=0, amp=False, drop_rate=0,
                  attn_drop_rate=0, drop_path_rate=0, label_smoothing=0)
    values.update(kwargs)
    return Config(**values).validate()


class PipelineTests(unittest.TestCase):
    def test_stratified_split_reproducible_disjoint(self):
        labels = [i for i in range(10) for _ in range(20)]
        train, val = stratified_split(labels, .1, 42)
        self.assertEqual((train, val), stratified_split(labels, .1, 42))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(200)))
        self.assertEqual([sum(labels[i] == c for i in val) for c in range(10)], [2]*10)

    def test_training_never_requests_test_dataset(self):
        fake = TensorDataset(torch.rand(40,3,32,32), torch.arange(40)%10)
        fake.targets = (torch.arange(40)%10).tolist()
        fake.classes = [str(i) for i in range(10)]
        fake.class_to_idx = {c:i for i,c in enumerate(fake.classes)}
        cfg = tiny_config()
        with patch('data.dataset', return_value=fake) as loader:
            training_loaders(cfg, 'unused', preprocessing(cfg))
        self.assertEqual([call.args[3] for call in loader.call_args_list], ['train', 'train'])

    def test_partial_accumulation_matches_full_batch(self):
        torch.manual_seed(7)
        x = torch.randn(5,4); y = torch.tensor([0,1,0,1,0])
        first = torch.nn.Linear(4,2); second = torch.nn.Linear(4,2)
        second.load_state_dict(first.state_dict())
        cfg = tiny_config(accumulation_steps=4, grad_clip=0)
        a = torch.optim.SGD(first.parameters(), lr=.1)
        b = torch.optim.SGD(second.parameters(), lr=.1)
        loss, updates = train_epoch(first, DataLoader(TensorDataset(x,y), batch_size=2), a,
                                   torch.amp.GradScaler('cuda', enabled=False), cfg, torch.device('cpu'))
        b.zero_grad(); torch.nn.functional.cross_entropy(second(x),y).backward(); b.step()
        self.assertEqual(updates,1)
        for p,q in zip(first.parameters(),second.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-7)

    def test_mixup_and_cutmix_only(self):
        cfg = tiny_config(mixup_alpha=0, cutmix_alpha=1, label_smoothing=.1)
        fn = mixing(cfg); self.assertIsNotNone(fn)
        images, targets = fn(torch.randn(2,3,32,32), torch.tensor([1,3]))
        torch.testing.assert_close(targets.sum(1), torch.ones(2))
        cfg.augment = False; self.assertIsNone(mixing(cfg))

    def test_cnn_learning_rate_groups_and_input_stem(self):
        cfg = Config(model_name='cnn', img_size=224, patch_size=16)
        model = build_model(cfg, initialize_pretrained=False)
        self.assertEqual(model.conv1.kernel_size,(7,7))
        groups = parameter_groups(model,cfg)
        self.assertEqual({id(p) for p in groups[1]['params']},{id(p) for p in model.fc.parameters()})
        self.assertEqual(len({id(p) for g in groups for p in g['params']}),len(list(model.parameters())))
        self.assertAlmostEqual(groups[0]['lr']/groups[1]['lr'], .1)
        del model
        scratch = build_model(Config())
        self.assertEqual(scratch.conv1.kernel_size,(3,3))

    def test_pretrained_failure_is_not_silenced(self):
        cfg = Config(model_name='vit_small',img_size=224,pretrained_id='vit_small_patch16_224.augreg_in21k_ft_in1k')
        with patch('models.factory.timm.create_model',side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError,'offline'): build_model(cfg)

    def test_metrics_include_absent_classes(self):
        result = classification_metrics([0,1],[0,0],10)
        self.assertEqual(result['accuracy'],50.)
        self.assertEqual(len(result['confusion_matrix']),10)

    def test_full_training_checkpoint_and_prediction(self):
        cfg = tiny_config()
        x = torch.rand(4,3,32,32); y = torch.tensor([0,1,2,3])
        loader = DataLoader(TensorDataset(x,y),batch_size=2)
        classes = [str(i) for i in range(10)]
        manifest = {'class_to_idx':{c:i for i,c in enumerate(classes)},'train_indices':[0,1], 'validation_indices':[2,3]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            with patch('train.training_loaders', return_value=(loader,loader,classes,manifest)):
                run(cfg,'unused',out,torch.device('cpu'))
            self.assertTrue((out/'best.pt').exists()); self.assertTrue((out/'last.pt').exists())
            engine = Predictor(out/'best.pt')
            result = engine.predict(Image.new('RGB',(70,40),'red'))
            self.assertEqual(len(result['all_classes']),10)
            self.assertAlmostEqual(sum(result['all_confidences']),100.,places=4)
            checkpoint = torch.load(out/'best.pt',weights_only=True)
            expected = transform(checkpoint['preprocessing'])(Image.new('RGB',(70,40),'red'))
            torch.testing.assert_close(expected, engine.transform(Image.new('RGB',(70,40),'red')))
            checkpoint['state_dict'].pop('head.weight')
            torch.save(checkpoint,out/'bad.pt')
            with self.assertRaises(RuntimeError): Predictor(out/'bad.pt')
            registry = Path(tmp)/'models.json'
            registry.write_text(json.dumps({'test':{'checkpoint':'run/best.pt','dataset':'cifar10'}}))
            client = Client(create_app(registry), Response)
            self.assertEqual(client.get('/').status_code,200)
            self.assertEqual(len(client.get('/models').json),1)
            buffer = io.BytesIO(); Image.new('RGB',(32,32),'red').save(buffer,format='PNG'); buffer.seek(0)
            response = client.post('/predict',data={'model':'test','file':(buffer,'sample.png')})
            self.assertEqual(response.status_code,200)
            self.assertEqual(client.post('/predict',data={'model':'test','file':(io.BytesIO(b'bad'),'fake.png')}).status_code,400)
            self.assertEqual(client.post('/predict',data={'model':'unknown'}).status_code,400)

    def test_unconfigured_demo_has_honest_empty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Client(create_app(Path(tmp)/'absent.json'), Response)
            self.assertEqual(client.get('/models').json,[])
            self.assertEqual(client.get('/').status_code,200)


if __name__ == '__main__': unittest.main()

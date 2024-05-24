import hydra
import torchvision.transforms as t
from omegaconf import DictConfig
from torchvision.models import resnet18

from oml.interfaces.models import IExtractor
from oml.lightning.pipelines.train import extractor_training_pipeline
from oml.registry.models import EXTRACTORS_REGISTRY
from oml.registry.transforms import TRANSFORMS_REGISTRY

from timm import create_model
from fastai.vision.learner import _update_first_layer, has_pool_type, create_head, num_features_model
import torch.nn as nn
from classification.ml_decoder.ml_decoder.ml_decoder import MLDecoder
from utils import get_model_size


def create_timm_body(arch:str, pretrained=False, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")


class TimmExtractor(IExtractor):

  def __init__(self, model_name, config={}):
    super().__init__()
    self.config = config
    self.config.setdefault('in_channels', 1)
    self.config.setdefault('pretrained', False)
    self.config.setdefault('embed_dim', 512)
    self.config.setdefault('decoder_embedding', 768)
    self.config.setdefault('no_head', False)
    self.body = create_timm_body(model_name, pretrained=self.config['pretrained'],
                                 n_in=self.config['in_channels'])
    nf = num_features_model(self.body)
    
    if self.config['no_head']:
      self.ml_decoder_head = nn.Identity()
    else:
      self.ml_decoder_head = MLDecoder(
        self.config['embed_dim'], initial_num_features=nf, decoder_embedding=self.config['decoder_embedding'])

  def forward(self, x):
    return self.ml_decoder_head(self.body(x))

  # this property is obligatory for IExtractor
  @property
  def feat_dim(self):
    return self.config['embed_dim']
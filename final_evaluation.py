import copy
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
import os

def capture_depth_buffer(test_dir, output_dir):
  img_width, img_height = (64, 64)

  renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
  renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))

  stldir = os.path.join(test_dir, 'database')
  for dirpath, dirnames, filenames in os.walk(stldir):
    os.makedirs(os.path.join(output_dir, 'pairs'), exist_ok=True)
    for stlname in tqdm(filenames):
      if 'stl' not in stlname: continue
      target = os.path.join(output_dir, 'pairs', stlname.replace('stl', 'png'))
      infile = os.path.join(stldir, stlname)
      if not os.path.isfile(infile): assert False, "Wrong stl path: %s" % (infile)
      renderer_pc.scene.clear_geometry()
      pcd = o3d.io.read_triangle_mesh(infile)

      mat = o3d.visualization.rendering.MaterialRecord()
      mat.shader = 'defaultUnlit'

      renderer_pc.scene.add_geometry("pcd", pcd, mat)

      # Optionally set the camera field of view (to zoom in a bit)
      vertical_field_of_view = 15.0  # between 5 and 90 degrees
      aspect_ratio = img_width / img_height  # azimuth over elevation
      near_plane = 0.1
      far_plane = 50.0
      fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
      renderer_pc.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

      # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
      center = [0, 0, 0]  # look_at target
      eye = [0, 0, 4]  # camera position
      up = [0, 1, 0]  # camera orientation
      renderer_pc.scene.camera.look_at(center, eye, up)

      depth_image = np.asarray(renderer_pc.render_to_depth_image())
      try:
        depth_image[depth_image==1.0] = np.unique(depth_image)[-2]
      except:
        plt.imshow(depth_image, cmap='gray')
        assert False

      normalized_image = depth_image.copy()
      normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())
      normalized_image = np.round(normalized_image, decimals=2)
      threshold = 0.2
      # print(threshold)
      # normalized_image[normalized_image < threshold] = 0.0
      normalized_image[normalized_image > threshold] =  threshold
      normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())
      normalized_image **= 2
      # plt.imshow(normalized_image, cmap='gray')
      plt.imsave(target, normalized_image, cmap='gray')
      


from retrieval.oml.timm_extractor import TimmExtractor, create_timm_body
from classification.ml_decoder.ml_decoder.ml_decoder import MLDecoder
from fastai.vision.learner import _update_first_layer, has_pool_type, create_head, num_features_model
import torch.nn as nn
from copy import deepcopy
from torch.optim import AdamW, SGD
from lion_pytorch import Lion
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import torch
import pandas as pd
from retrieval.transform import test_transform


def image_retrieval(args, test_dir, output_dir):
  config = dict(args._get_kwargs())
  extractor = TimmExtractor(args.model_name, config)
  
  if not os.path.isfile(os.path.join(args.save_path, 'pytorch_model.bin')):
    assert False, "Model does not exist."
    
  best_score, state_dict = torch.load(os.path.join(args.save_path, 'pytorch_model.bin'))
  extractor.load_state_dict(state_dict)
  print('Loaded pretrained model with score=%.4f' % best_score)
  
  db_img_path = os.path.join(test_dir, 'database/pairs')
  q_img_path = os.path.join(test_dir, 'queries')
  n_db = len(os.listdir(db_img_path))
  n_q = len(os.listdir(q_img_path))
  
  eval_model = extractor
  db_vecs = []
  eval_model.to(args.device)

  for i in tqdm(range(n_db), position=0, leave=True):
    db_img = cv2.imread(os.path.join(db_img_path, '%d.png' % i))
    x = test_transform(image=db_img)['image'][[0], :, :]
    x = x.to(args.device)[None]
    eval_model.eval()
    with torch.no_grad():
      db_vec = eval_model(x)
    db_vec = db_vec.detach().cpu().numpy()
    db_vecs += [db_vec]

  def score_query(q_vec):
    db_scores = []
    for i in range(n_db):
      db_vec = db_vecs[i]
      db_scores += [-((db_vec - q_vec) ** 2).sum()]
    return db_scores
    

  def retrieve():
    res = []
    for i in tqdm(range(n_q), position=0, leave=True):
      img_name = os.path.join(q_img_path, '%d.png' % i)
      q_img = cv2.imread(img_name)
      x = test_transform(image=q_img)['image'][[0], :, :]
      x = x.to(args.device)[None]
      eval_model.eval()
      with torch.no_grad():
        q_vec = eval_model(x)
      q_vec = q_vec.detach().cpu().numpy()
      db_scores = score_query(q_vec)
      rank = np.argsort(db_scores)[::-1]
      res += [{
        'query_name': img_name.split('/')[-1],
        'predictions': ','.join(['%d.stl' % i for i in rank[:5]]),
      }]
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(output_dir, 'pred.csv'), index=False)

  retrieve()
      
      
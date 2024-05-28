import cv2
from tqdm import tqdm
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import shutil


# Convert character c to integer for the label
def char_to_int(c):
  # Convert character c to bytes
  res = c.encode()
  # Convert bytes array to integer using big endian
  res = int.from_bytes(res, 'big')
  return res

# Convert int to character
def int_to_char(x):
  # Convert int x to bytes array of len=4, then decode to string
  res = x.to_bytes(4, 'big').decode()
  # Bytes array might have excessive 0 padding => remove them
  res = res.replace('\x00', '')
  return res

# Read all the patches transcripts
with open('input/nomnaocr/Patches/All.txt', encoding="utf8") as f:
  patches_data = f.readlines()

# Get all the characters
charset = set()
for i in range(len(patches_data)):
  # Transcripts are in format of (image_path, text transcripts)
  img_path, label = patches_data[i].split('\t')
  label = label.strip()
  for c in label:
    if c == '?':
      # Skip ? characters
      continue
    charset.add(c)

# Sort for consistency
charset = sorted(list(charset))


import random
empty_img = np.zeros((48, 48, 3), dtype=np.uint8)

# Draw bold character to image
def draw_text_bold(xy, text, font, fill=(0,0,0)):
  x, y = xy
  draw.text((x-1, y), text, font=font, fill=fill)
  draw.text((x, y-1), text, font=font, fill=fill)
  draw.text((x-1, y-1), text, font=font, fill=fill)
  draw.text((x, y), text, font=font, fill=fill)

# Save image
def save_img(img, cls_name, font, bold):
  path = os.path.join(output_dir, cls_name)
  os.makedirs(os.path.join(path), exist_ok=True)
  font_name = '_'.join(font.getname()).replace(' ', '_')
  filename = os.path.join(path, '%s_%d%s.png' % (font_name, font.size, "_bold" if bold else ""))
  assert cv2.imwrite(filename, img), "WRITE FAILED"
  
# TTFont library for checking if character is in the font.
from fontTools.ttLib import TTFont

# Copied from stackoverflow
def char_in_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False
  
# Read all fonts from folder
font_list = [
  *[(ImageFont.truetype(os.path.join('input/fonts', p), 32),
    (os.path.join('input/fonts', p), 32)) for p in os.listdir('input/fonts')]
]


output_dir = 'input/my_classify_data/'
for char in tqdm(charset):
  curr_lbl = char_to_int(char)
  empty_img = np.zeros((48, 48, 3), dtype=np.uint8)
  
  path = os.path.join(output_dir, str(curr_lbl))
  # Remove the folder if exist => Redraw all images
  shutil.rmtree(os.path.join(path), ignore_errors=True)
  for font, (url, size) in font_list:
    if not char_in_font(char, TTFont(url)):
      continue
    
    try:
      # Empty white image
      img_pil = Image.fromarray(255-empty_img.copy())
      draw = ImageDraw.Draw(img_pil)
      _, _, w, h = draw.textbbox((0, 0), char, font)
      draw.text(((24-w//2),(24-h//2)), char, (0, 0, 0), font)
      img = np.array(img_pil)

      if (img < 255).sum() < 10:
        # Image too white (Virtually nothing has been drawn)
        continue
      save_img(img, str(curr_lbl), font, False)
    
      img_pil = Image.fromarray(255-empty_img.copy())
      draw = ImageDraw.Draw(img_pil)
      _, _, w, h = draw.textbbox((0, 0), char, font)
      draw_text_bold(((24-w//2),(24-h//2)), char, font)
      img = np.array(img_pil)
      
      if (img < 255).sum() < 10:
        # Image too white (Virtually nothing has been drawn)
        continue
      
      save_img(img, str(curr_lbl), font, True)
    except: continue
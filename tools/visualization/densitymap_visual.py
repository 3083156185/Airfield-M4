import os, sys, json, tqdm, cv2, shutil
import argparse
import numpy as np
sys.path.append('./')

from engine.deim.utils import DensityMapGenerator, visualize_density_map

SAVE_PATH = 'DensityMap'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--json', '-j', required=True, type=str)
    args = parser.parse_args()

    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    os.makedirs(SAVE_PATH)

    with open(args.json) as f:
        data = json.load(f)

    image_box_dict = {}
    image_size_dict = {}
    for image_data in data['images']:
        image_box_dict[image_data['id']] = []
        image_size_dict[image_data['id']] = [image_data['file_name'], image_data['height'], image_data['width']]
    
    for box_data in data['annotations']:
        box = np.array(box_data['bbox'])
        box[:2] = box[:2] + box[2:] / 2
        image_box_dict[box_data['image_id']].append(box)
    
    for image_key in tqdm.tqdm(image_box_dict):
        image = cv2.imread(os.path.join(args.image_path, image_size_dict[image_key][0]))
        dmg = DensityMapGenerator(image_size_dict[image_key][1:])
        densitymap = dmg.generate_from_boxes(np.array(image_box_dict[image_key]), method='gaussian')
        visualize_density_map(image, densitymap, image_box_dict[image_key], os.path.join(SAVE_PATH, image_size_dict[image_key][0]))

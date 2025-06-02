import json
import os
from PIL import Image

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    if not os.path.isfile(coco_json_path):
        print(f"❌ File not found: {coco_json_path}")
        return

    if os.path.getsize(coco_json_path) == 0:
        print(f"⚠️ JSON file is empty: {coco_json_path}")
        return

    try:
        with open(coco_json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in {coco_json_path}: {e}")
        return

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']
        category_id = ann['category_id']
        image_info = images[img_id]
        image_path = os.path.join(images_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w = bbox[2] / width
        h = bbox[3] / height

        annotation_line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        txt_filename = os.path.splitext(image_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'a') as txt_file:
            txt_file.write(annotation_line)

    print(f"✅ Conversion complete for: {coco_json_path}")

if __name__ == "__main__":
    sets = ['train', 'val', 'test']
    base_path = 'image_dataset'

    for split in sets:
        coco_json_path = os.path.join(base_path, split, f"{split}.json")
        images_dir = os.path.join(base_path, split, "img")
        output_dir = os.path.join(base_path, split, "labels")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🔍 Processing {split} set...")
        convert_coco_to_yolo(coco_json_path, images_dir, output_dir)

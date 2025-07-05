import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

def extract_data_from_xml(root_dir):
    xml_path = os.path.join(root_dir, 'words.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall('taggedRectangles'):
            for bb in bbs:
                # Check non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue

                if ' ' in bb[0].text:
                    continue

                bbs_of_img.append([
                    float(bb.attrib['x']),
                    float(bb.attrib['y']),
                    float(bb.attrib['width']),
                    float(bb.attrib['height'])
                ])
                labels_of_img.append(bb[0].text.lower())

        if bbs_of_img:  # Only add if there are valid bboxes
            img_path = os.path.join(root_dir, img[0].text)
            img_paths.append(img_path)
            img_sizes.append([
                int(img[1].attrib['x']),
                int(img[1].attrib['y'])
            ])
            bboxes.append(bbs_of_img)
            img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes

def convert_to_yolo_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []

    for image_path, image_size, bboxes in zip(image_paths, image_sizes, bounding_boxes):
        image_width, image_height = image_size

        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # Calculate normalized bounding box coordinates
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Because we only have one class, we set class_id to 0
            class_id = 0

            # Convert to YOLO format
            yolo_label = f"{class_id} {center_x} {center_y} {normalized_width} {normalized_height}"
            yolo_labels.append(yolo_label)

        yolo_data.append((image_path, yolo_labels))

    return yolo_data

def save_data(data, src_img_dir, save_dir):
    # Create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Make images and labels folder
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, yolo_labels in data:
        # Copy image to images folder
        shutil.copy(
            image_path, os.path.join(save_dir, "images")
        )

        # Save labels to labels folder
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        with open(os.path.join(save_dir, "labels", f"{image_name}.txt"), "w") as f:
            for label in yolo_labels:
                f.write(f"{label}\n")

dataset_dir = 'icdar2003/SceneTrialTrain'
img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(dataset_dir)

# Define class labels
class_labels = ["text"]

# Convert data into YOLO format
yolo_data = convert_to_yolo_format(img_paths, img_sizes, bboxes)

seed = 0
val_size = 0.2
test_size = 0.125
is_shuffle = True

train_data, test_data = train_test_split(
    yolo_data,
    test_size=val_size,
    random_state=seed,
    shuffle=is_shuffle,
)

test_data, val_data = train_test_split(
    test_data,
    test_size=test_size,
    random_state=seed,
    shuffle=is_shuffle,
)

save_yolo_data_dir = "datasets/yolo_data"
os.makedirs(save_yolo_data_dir, exist_ok=True)
save_train_dir = os.path.join(save_yolo_data_dir, "train")
save_val_dir = os.path.join(save_yolo_data_dir, "val")
save_test_dir = os.path.join(save_yolo_data_dir, "test")

save_data(train_data, dataset_dir, save_train_dir)
save_data(test_data, dataset_dir, save_val_dir)
save_data(val_data, dataset_dir, save_test_dir)

print(f"Saved {len(train_data)} training images")
print(f"Saved {len(val_data)} validation images") 
print(f"Saved {len(test_data)} test images")

# Create data.yaml for YOLOv11
yaml_content = f"""train: {os.path.abspath(os.path.join(save_train_dir, 'images'))}
val: {os.path.abspath(os.path.join(save_val_dir, 'images'))}
test: {os.path.abspath(os.path.join(save_test_dir, 'images'))}

nc: 1
names: ['text']
"""

with open(os.path.join(save_yolo_data_dir, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("YOLO dataset preparation completed!")
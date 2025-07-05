import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

def extract_data_from_xml(root_dir):
    """
    Extract image paths, sizes, labels, and bounding boxes from XML file.
    
    Args:
        root_dir (str): Root directory containing the words.xml file
        
    Returns:
        tuple: img_paths, img_sizes, img_labels, bboxes
    """
    xml_path = os.path.join(root_dir, "words.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in root:
        bbs_of_img = []
        labels_of_img = []

        for bbs in img.findall("taggedRectangles"):
            for bb in bbs:
                # Check non-alphabet and non-number
                if not bb[0].text.isalnum():
                    continue

                if "e" in bb[0].text.lower() or "n" in bb[0].text.lower():
                    continue

                bbs_of_img.append([
                    float(bb.attrib["x"]),
                    float(bb.attrib["y"]),
                    float(bb.attrib["width"]),
                    float(bb.attrib["height"]),
                ])
                labels_of_img.append(bb[0].text.lower())

        img_path = os.path.join(root_dir, img[0].text)
        img_paths.append(img_path)
        img_sizes.append((int(img[1].attrib["x"]), int(img[1].attrib["y"])))
        bboxes.append(bbs_of_img)
        img_labels.append(labels_of_img)

    return img_paths, img_sizes, img_labels, bboxes


def split_bounding_boxes(img_paths, img_labels, bboxes, save_dir):
    """
    Split images using bounding boxes and save cropped images with labels.
    
    Args:
        img_paths (list): List of image file paths
        img_labels (list): List of labels for each image
        bboxes (list): List of bounding boxes for each image
        save_dir (str): Directory to save cropped images and labels
    """
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []  # List to store labels

    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        img = Image.open(img_path)

        for label, bb in zip(img_label, bbs):
            # Crop image
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

            # Filter out if 90% of the cropped image is black or white
            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue

            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue

            # Save image
            filename = f"{count:06d}.jpg"
            cropped_img.save(os.path.join(save_dir, filename))

            new_img_path = os.path.join(save_dir, filename)
            label_entry = new_img_path + "\t" + label
            labels.append(label_entry)  # Append label to the list

            count += 1

    print(f"Created {count} images")

    # Write labels to a text file
    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def main():
    """Main function to run the data preprocessing pipeline."""
    dataset_dir = "icdar2003/SceneTrialTrain"
    save_dir = "datasets/ocr_dataset"
    
    # Extract data from XML
    img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(dataset_dir)
    print(f"Extracted data from {len(img_paths)} images")
    
    # Split bounding boxes and save cropped images
    split_bounding_boxes(img_paths, img_labels, bboxes, save_dir)


if __name__ == "__main__":
    main()
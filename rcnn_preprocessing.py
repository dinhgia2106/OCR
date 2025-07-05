import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split

def load_data(root_dir):
    """
    Load image paths and labels from the processed dataset.
    
    Args:
        root_dir (str): Directory containing the labels.txt file
        
    Returns:
        tuple: (img_paths, labels) lists
    """
    img_paths = []
    labels = []
    
    # Read labels from text file
    with open(os.path.join(root_dir, "labels.txt"), "r") as f:
        for label in f:
            labels.append(label.strip().split("\t")[1])
            img_paths.append(label.strip().split("\t")[0])
    
    print(f"Total images: {len(img_paths)}")
    return img_paths, labels


def build_vocabulary(labels):
    """
    Build vocabulary from all characters in the labels.
    
    Args:
        labels (list): List of text labels
        
    Returns:
        tuple: (chars, vocab_size, char_to_idx, idx_to_char, max_label_len)
    """
    letters = [char.split(".")[0].lower() for char in labels]
    letters = "".join(letters)
    letters = sorted(list(set(list(letters))))
    
    # Create a string of all characters in the dataset
    chars = "".join(letters)
    
    # For "blank" character
    blank_char = "-"
    chars += blank_char
    vocab_size = len(chars)
    
    print(f"Vocab: {chars}")
    print(f"Vocab size: {vocab_size}")
    
    # Create character to index and index to character mappings
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    
    # Find maximum label length for padding
    max_label_len = max([len(label) for label in labels])
    
    return chars, vocab_size, char_to_idx, idx_to_char, max_label_len


def encode(label, char_to_idx, max_label_len):
    """
    Encode a text label into tensor of character indices with padding.
    
    Args:
        label (str): Text label to encode
        char_to_idx (dict): Character to index mapping
        max_label_len (int): Maximum label length for padding
        
    Returns:
        tuple: (padded_labels, lengths) as tensors
    """
    encoded_labels = torch.tensor(
        [char_to_idx[char] for char in label],
        dtype=torch.int32
    )
    label_len = len(encoded_labels)
    lengths = torch.tensor(
        label_len,
        dtype=torch.int32
    )
    padded_labels = F.pad(
        encoded_labels,
        (0, max_label_len - label_len),
        value=0
    )
    
    return padded_labels, lengths


def decode(encoded_sequences, idx_to_char, blank_char="-"):
    """
    Decode tensor sequences back to text strings.
    
    Args:
        encoded_sequences (torch.Tensor): Encoded sequences
        idx_to_char (dict): Index to character mapping
        blank_char (str): Blank character for CTC
        
    Returns:
        list: Decoded text sequences
    """
    decoded_sequences = []
    
    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None  # To track the previous character
        
        for token in seq:
            if token != 0:  # Ignore padding (token = 0)
                char = idx_to_char[token.item()]
                # Append the character if it's not a blank or the same as the previous character
                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append(char)
                prev_char = char  # Update previous character
        
        decoded_sequences.append("".join(decoded_label))
    
    print(f"From {encoded_sequences} to {decoded_sequences}")
    
    return decoded_sequences


def get_data_transforms():
    """
    Define data transformations for training and validation.
    
    Returns:
        dict: Dictionary containing train and val transforms
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(
                degrees=1,
                shear=1,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.5,
                interpolation=3,
            ),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    }
    
    return data_transforms


def split_dataset(img_paths, labels, seed=0, val_size=0.1, test_size=0.1, is_shuffle=True):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        img_paths (list): List of image paths
        labels (list): List of labels
        seed (int): Random seed for reproducibility
        val_size (float): Validation set size ratio
        test_size (float): Test set size ratio
        is_shuffle (bool): Whether to shuffle the data
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate validation set
    X_train, X_val, y_train, y_val = train_test_split(
        img_paths,
        labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle,
    )
    
    # Second split: separate test set from remaining training data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle,
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main function to run the RCNN preprocessing pipeline."""
    # Set the root directory
    root_dir = "datasets/ocr_dataset"
    
    # Load data
    img_paths, labels = load_data(root_dir)
    
    # Build vocabulary
    chars, vocab_size, char_to_idx, idx_to_char, max_label_len = build_vocabulary(labels)
    
    # Get data transforms
    data_transforms = get_data_transforms()
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(img_paths, labels)
    
    # Example of encoding and decoding
    if labels:
        example_label = labels[0]
        print(f"\nExample encoding/decoding:")
        print(f"Original label: {example_label}")
        
        padded_label, length = encode(example_label, char_to_idx, max_label_len)
        print(f"Encoded: {padded_label}")
        print(f"Length: {length}")
        
        decoded = decode([padded_label], idx_to_char)
        print(f"Decoded: {decoded[0]}")
    
    return {
        'img_paths': img_paths,
        'labels': labels,
        'chars': chars,
        'vocab_size': vocab_size,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'max_label_len': max_label_len,
        'data_transforms': data_transforms,
        'train_data': (X_train, y_train),
        'val_data': (X_val, y_val),
        'test_data': (X_test, y_test)
    }


if __name__ == "__main__":
    preprocessing_data = main()
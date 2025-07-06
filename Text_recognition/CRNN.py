import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import time
from rcnn_preprocessing import main as get_preprocessing_data, encode
import os
from tqdm import tqdm

class STRDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        char_to_idx,
        max_label_len,
        label_encoder=None,
        transform=None,
    ):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(
                label, self.char_to_idx, self.max_label_len
            )
            return img, encoded_label, label_len

        return img, label


class CRNN(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        # Remove autocast from forward - it will be handled in training loop
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten the feature map
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # Based on CTC

        return x


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The CRNN model
        dataloader: DataLoader for evaluation
        criterion: Loss function (CTCLoss)
        device: Device to run evaluation on
        
    Returns:
        float: Average loss
    """
    model.eval()
    losses = []
    with torch.no_grad():
        # Add tqdm for evaluation progress
        eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, labels, labels_len in eval_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            outputs = model(inputs)
            logits_lens = torch.full(
                size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long
            ).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            losses.append(loss.item())
            
            # Update progress bar
            eval_pbar.set_postfix({'loss': loss.item()})

    loss = sum(losses) / len(losses)
    return loss


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_dir, 
                   vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob, 
                   is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch
        train_losses: Training losses history
        val_losses: Validation losses history
        save_dir: Directory to save the checkpoint
        vocab_size: Vocabulary size
        char_to_idx: Character to index mapping
        chars: Characters string
        hidden_size: Hidden size of the model
        n_layers: Number of layers
        dropout_prob: Dropout probability
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'char_to_idx': char_to_idx,
        'chars': chars,
        'hidden_size': hidden_size,
        'n_layers': n_layers,
        'dropout_prob': dropout_prob,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    if is_best:
        save_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, save_path)
        print(f"Best model saved at epoch {epoch + 1}")
    else:
        save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")


def fit(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs,
    save_dir, vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob,
    scaler, patience=7, save_every=10
):
    """
    Train the model with early stopping and periodic saving.
    
    Args:
        model: The CRNN model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs
        save_dir: Directory to save checkpoints
        vocab_size: Vocabulary size
        char_to_idx: Character to index mapping
        chars: Characters string
        hidden_size: Hidden size of the model
        n_layers: Number of layers
        dropout_prob: Dropout probability
        scaler: GradScaler for mixed precision training
        patience: Early stopping patience
        save_every: Save checkpoint every N epochs
        
    Returns:
        tuple: (train_losses, val_losses)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        start = time.time()
        batch_train_losses = []

        model.train()
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for idx, (inputs, labels, labels_len) in enumerate(train_pbar):
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            with autocast():
                outputs = model(inputs)

                logits_lens = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long,
                ).to(device)

                loss = criterion(outputs, labels, logits_lens, labels_len)

            # Scale loss before backward
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # Step optimizer through scaler
            scaler.step(optimizer)
            scaler.update()

            batch_train_losses.append(loss.item())
            
            # Update training progress bar
            train_pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': sum(batch_train_losses) / len(batch_train_losses)
            })

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update main progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'best_val': f'{best_val_loss:.4f}',
            'patience': f'{patience_counter}/{patience}'
        })

        print(
            f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\t\tTime: {time.time() - start:.2f} seconds"
        )

        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(
                model, optimizer, epoch, train_losses, val_losses, save_dir,
                vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob,
                is_best=True
            )
        else:
            patience_counter += 1

        # Save checkpoint every save_every epochs
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_losses, val_losses, save_dir,
                vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob,
                is_best=False
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        scheduler.step()

    return train_losses, val_losses


def create_dataloaders(preprocessing_data):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        preprocessing_data: Dictionary containing preprocessed data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    X_train, y_train = preprocessing_data['train_data']
    X_val, y_val = preprocessing_data['val_data']
    X_test, y_test = preprocessing_data['test_data']
    
    char_to_idx = preprocessing_data['char_to_idx']
    max_label_len = preprocessing_data['max_label_len']
    data_transforms = preprocessing_data['data_transforms']

    train_dataset = STRDataset(
        X_train,
        y_train,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["train"],
    )
    val_dataset = STRDataset(
        X_val,
        y_val,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )
    test_dataset = STRDataset(
        X_test,
        y_test,
        char_to_idx=char_to_idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )

    train_batch_size = 64
    test_batch_size = 64 * 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def main():
    """Main function to run the CRNN training pipeline."""
    # Get preprocessed data
    preprocessing_data = get_preprocessing_data()
    
    # Extract necessary components
    vocab_size = preprocessing_data['vocab_size']
    char_to_idx = preprocessing_data['char_to_idx']
    chars = preprocessing_data['chars']
    blank_char = "-"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(preprocessing_data)
    
    # Model parameters
    hidden_size = 256
    n_layers = 3
    dropout_prob = 0.2
    unfreeze_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    # Create model
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout_prob,
        unfreeze_layers=unfreeze_layers,
    ).to(device)

    # Training parameters
    epochs = 100
    lr = 5e-5
    weight_decay = 1e-5
    scheduler_step_size = int(epochs * 0.5)
    patience = 7
    save_every = 10

    # Create save directory
    save_dir = "crnn"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Loss, optimizer, and scheduler
    criterion = nn.CTCLoss(
        blank=char_to_idx[blank_char],
        zero_infinity=True,
        reduction="mean",
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.1
    )

    print("Starting training...")
    # Training with early stopping and periodic saving
    train_losses, val_losses = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs,
        save_dir, vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob,
        scaler, patience=patience, save_every=save_every
    )

    # Evaluation
    print("\nEvaluating final model...")
    val_loss = evaluate(model, val_loader, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    print("Evaluation on val/test dataset")
    print("Val loss:", val_loss)
    print("Test loss:", test_loss)

    # Save final model
    save_checkpoint(
        model, optimizer, len(train_losses)-1, train_losses, val_losses, save_dir,
        vocab_size, char_to_idx, chars, hidden_size, n_layers, dropout_prob,
        is_best=False
    )

    print(f"Training completed. Models saved in {save_dir}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    model, train_losses, val_losses = main()
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# =========================================
# 1. DATA PREPARATION AND DATASET CLASSES
# =========================================

class PretrainDataset(torch.utils.data.Dataset):
    """Dataset for self-supervised pretraining (MSM) using PNG images"""
    def __init__(self, spectrogram_dir):
        self.spectrogram_dir = spectrogram_dir
        self.file_list = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load PNG spectrogram
        img_path = os.path.join(self.spectrogram_dir, self.file_list[idx])
        spec = Image.open(img_path).convert('L')  # Convert to grayscale
        spec = np.array(spec, dtype=np.float32)

        # Normalize to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

        # Add channel dimension and create radio sentence (16 tokens of 256x16)
        tokens = []
        for i in range(0, 256, 16):
            token = spec[:, i:i+16]  # (256, 16)
            tokens.append(token)

        # Stack tokens and add channel dimension
        sentence = np.stack(tokens, axis=0)  # (16, 256, 16)
        sentence = np.expand_dims(sentence, 1)  # (16, 1, 256, 16)

        return torch.tensor(sentence, dtype=torch.float32)

class SegmentationDataset(torch.utils.data.Dataset):
    """Dataset for segmentation task using PNG images"""
    def __init__(self, spectrogram_dir, label_dir, label_mode='grayscale'):
        """
        Args:
            spectrogram_dir: Directory with spectrogram PNGs
            label_dir: Directory with label PNGs
            label_mode: 'grayscale' for single-channel 0,1,2 labels
                        or 'rgb' for color-coded labels
        """
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir
        self.file_list = [f for f in os.listdir(spectrogram_dir) if f.endswith('.png')]
        self.label_mode = label_mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load spectrogram PNG
        img_path = os.path.join(self.spectrogram_dir, self.file_list[idx])
        spec = Image.open(img_path).convert('L')  # Grayscale
        spec = np.array(spec, dtype=np.float32)

        # Normalize spectrogram
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

        # Load label PNG
        label_path = os.path.join(self.label_dir, self.file_list[idx])
        if self.label_mode == 'grayscale':
            label_img = Image.open(label_path).convert('L')
            label = np.array(label_img, dtype=np.uint8)
        else:  # RGB mode
            label_img = Image.open(label_path).convert('RGB')
            label = np.array(label_img)
            # Convert RGB to class labels:
            #   Black (0,0,0) -> 0 (Noise)
            #   Red (255,0,0) -> 1 (NR)
            #   Green (0,255,0) -> 2 (LTE)
            label = np.zeros(label.shape[:2], dtype=np.uint8)
            label[(label_img.getdata(0) == 255) & (label_img.getdata(1) == 0) & (label_img.getdata(2) == 0)] = 1
            label[(label_img.getdata(0) == 0) & (label_img.getdata(1) == 255) & (label_img.getdata(2) == 0)] = 2

        # Create spectrogram tokens
        tokens = []
        for i in range(0, 256, 16):
            token = spec[:, i:i+16]  # (256, 16)
            tokens.append(token)

        # Stack tokens and add channel dimension
        sentence = np.stack(tokens, axis=0)  # (16, 256, 16)
        sentence = np.expand_dims(sentence, 1)  # (16, 1, 256, 16)

        return (
            torch.tensor(sentence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# =========================================
# 2. MODEL ARCHITECTURES (AS IN SECTION III)
# =========================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + out_channels,
            4 * out_channels,
            kernel_size,
            padding=padding
        )
        self.out_channels = out_channels

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        i, f, o, g = torch.split(conv_output, self.out_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels
            self.layers.append(ConvLSTMCell(layer_in, out_channels, kernel_size))
        self.hidden_states = [None] * num_layers

    def initialize_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        for i in range(self.num_layers):
            self.hidden_states[i] = (
                torch.zeros(batch_size, self.layers[i].out_channels, height, width).to(device),
                torch.zeros(batch_size, self.layers[i].out_channels, height, width).to(device)
            )

    def forward(self, x):
        b, seq_len, _, h, w = x.size()
        self.initialize_hidden(b, h, w)
        outputs = []

        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            for layer_idx, layer in enumerate(self.layers):
                h_next, c_next = layer(input_t, self.hidden_states[layer_idx])
                self.hidden_states[layer_idx] = (h_next, c_next)
                input_t = h_next
            outputs.append(h_next)

        return torch.stack(outputs, dim=1)

class MSMPretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlstm = ConvLSTM(in_channels=1, out_channels=64, num_layers=5, kernel_size=3)
        self.conv3d = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.convlstm(x)  # (b, seq_len, 64, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (b, 64, seq_len, h, w)
        x = self.conv3d(x)
        x = self.relu(x)
        return x.permute(0, 2, 1, 3, 4)  # (b, seq_len, 1, h, w)

class ForecastingModel(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.convlstm = pretrained_model.convlstm
        for param in self.convlstm.parameters():
            param.requires_grad = False
        self.forecast_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.convlstm(x)  # (b, seq_len, 64, 256, 16)
        last_output = x[:, -1, :, :, :]  # (b, 64, 256, 16)
        return self.forecast_head(last_output)  # (b, 1, 256, 16)

class SegmentationModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=3):
        super().__init__()
        self.convlstm = pretrained_model.convlstm
        for param in self.convlstm.parameters():
            param.requires_grad = False
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.convlstm(x)  # (b, 16, 64, 256, 16)
        x = x.permute(0, 2, 1, 3, 4)  # (b, 64, 16, 256, 16)
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, h, t * w)  # (b, 64, 256, 256)
        return self.seg_head(x).permute(0, 2, 3, 1)  # (b, 256, 256, num_classes)

# =========================================
# 3. UTILITY FUNCTIONS
# =========================================

def apply_mask(sentence, mask_ratio=0.2):
    """Apply masking to spectrogram tokens (MSM)"""
    b, seq_len, c, h, w = sentence.shape
    masked_sentence = sentence.clone()
    mask = torch.rand(b, seq_len, device=sentence.device) < mask_ratio

    # Replace masked tokens with white noise
    for i in range(b):
        for j in range(seq_len):
            if mask[i, j]:
                noise = torch.randn_like(sentence[i, j])
                masked_sentence[i, j] = noise

    return masked_sentence, mask

def create_forecasting_data(sentence):
    """Create input-output pairs for forecasting task"""
    inputs = sentence[:, :-1]  # First T-1 tokens
    target = sentence[:, -1]   # Last token
    return inputs, target

def calculate_accuracy(output, target):
    """Calculate accuracy for segmentation task"""
    _, predicted = torch.max(output, dim=-1)
    correct = (predicted == target).sum().item()
    total = target.numel()
    return correct / total

# =========================================
# 4. TRAINING PIPELINE (SECTION III & IV)
# =========================================

def pretrain_msm(spectrogram_dir, batch_size=16, epochs=50, lr=1e-3):
    """Self-supervised pretraining with Masked Spectrogram Modeling"""
    # Prepare dataset
    dataset = PretrainDataset(spectrogram_dir)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize model
    model = MSMPretrainedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}"):
            sentence = batch.to(device)  # (b, 16, 1, 256, 16)

            # Apply masking
            masked_sentence, mask = apply_mask(sentence)

            # Forward pass
            reconstructed = model(masked_sentence)

            # Calculate loss only on masked tokens
            loss = 0
            for i in range(sentence.size(0)):
                for j in range(sentence.size(1)):
                    if mask[i, j]:
                        loss += criterion(reconstructed[i, j], sentence[i, j])

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sentence = batch.to(device)
                masked_sentence, mask = apply_mask(sentence)
                reconstructed = model(masked_sentence)

                loss = 0
                for i in range(sentence.size(0)):
                    for j in range(sentence.size(1)):
                        if mask[i, j]:
                            loss += criterion(reconstructed[i, j], sentence[i, j])
                val_loss += loss.item()

        # Print epoch statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_pretrained_model.pth")

    return model

def finetune_forecasting(pretrained_model, spectrogram_dir, batch_size=16, epochs=30, lr=1e-4):
    """Fine-tune for spectrum forecasting task"""
    # Prepare dataset
    dataset = PretrainDataset(spectrogram_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize forecasting model
    model = ForecastingModel(pretrained_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Forecast Finetune Epoch {epoch+1}"):
            sentence = batch.to(device)  # (b, 16, 1, 256, 16)

            # Create forecasting data
            inputs, target = create_forecasting_data(sentence)

            # Forward pass
            prediction = model(inputs)

            # Calculate loss
            loss = criterion(prediction, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sentence = batch.to(device)
                inputs, target = create_forecasting_data(sentence)
                prediction = model(inputs)
                loss = criterion(prediction, target)
                val_loss += loss.item()

        # Print epoch statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_forecasting_model.pth")

    return model

def finetune_segmentation(pretrained_model, spectrogram_dir, label_dir,
                          batch_size=8, epochs=50, lr=1e-4):
    """Fine-tune for spectrogram segmentation task"""
    # Prepare dataset
    dataset = SegmentationDataset(spectrogram_dir, label_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize segmentation model
    model = SegmentationModel(pretrained_model, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch in tqdm(train_loader, desc=f"Segmentation Finetune Epoch {epoch+1}"):
            sentence, labels = batch
            sentence = sentence.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sentence)

            # Calculate loss and accuracy
            loss = criterion(outputs.permute(0, 3, 1, 2), labels)
            acc = calculate_accuracy(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sentence, labels = batch
                sentence = sentence.to(device)
                labels = labels.to(device)

                outputs = model(sentence)
                loss = criterion(outputs.permute(0, 3, 1, 2), labels)
                acc = calculate_accuracy(outputs, labels)

                val_loss += loss.item()
                val_acc += acc

        # Print epoch statistics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_segmentation_model.pth")

    return model

# =========================================
# 5. EVALUATION METRICS (SECTION IV)
# =========================================

def evaluate_forecasting(model, test_dir):
    """Evaluate forecasting model using resource grid metric"""
    dataset = PretrainDataset(test_dir)
    loader = DataLoader(dataset, batch_size=16)

    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            sentence = batch.to(device)
            inputs, target = create_forecasting_data(sentence)
            prediction = model(inputs)

            # Convert to resource grid (binary occupancy)
            target_grid = convert_to_resource_grid(target)
            pred_grid = convert_to_resource_grid(prediction)

            all_targets.append(target_grid)
            all_predictions.append(pred_grid)

    # Calculate probability of correct occupied predictions
    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)

    occupied_mask = (targets == 1)
    correct_occupied = (predictions[occupied_mask] == targets[occupied_mask]).float().mean()

    print(f"Probability of correct occupied predictions: {correct_occupied:.4f}")
    return correct_occupied

def convert_to_resource_grid(spectrogram, time_res=1, freq_res=5):
    """
    Convert spectrogram to binary resource grid
    (Simplified implementation based on Section IV-A)
    """
    # Dummy implementation - real implementation would use actual thresholds
    # and resource block calculations as in Equation (4)
    mean = spectrogram.mean(dim=(1, 2, 3), keepdim=True)
    std = spectrogram.std(dim=(1, 2, 3), keepdim=True)
    threshold = mean + 0.5 * std
    return (spectrogram > threshold).float()

def evaluate_segmentation(model, test_spectrogram_dir, test_label_dir):
    """Evaluate segmentation model using confusion matrix"""
    dataset = SegmentationDataset(test_spectrogram_dir, test_label_dir)
    loader = DataLoader(dataset, batch_size=8)

    model.eval()
    confusion_matrix = torch.zeros(3, 3)  # Classes: Noise, NR, LTE

    with torch.no_grad():
        for sentence, labels in loader:
            sentence = sentence.to(device)
            labels = labels.cpu()

            outputs = model(sentence)
            _, predicted = torch.max(outputs.cpu(), dim=-1)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t, p] += 1

    # Normalize and print results
    confusion_matrix /= confusion_matrix.sum(dim=1, keepdim=True)
    print("Confusion Matrix (Rows: True, Columns: Predicted):")
    print(confusion_matrix)
    return confusion_matrix

# =========================================
# 6. MAIN EXECUTION PIPELINE
# =========================================

if __name__ == "__main__":
    # Configuration
    PRETRAIN_DIR = "data/pretrain_spectrograms"
    FORECAST_DIR = "data/forecast_spectrograms"
    SEGMENT_SPECT_DIR = "data/segment_spectrograms"
    SEGMENT_LABEL_DIR = "data/segment_labels"
    TEST_FORECAST_DIR = "data/test_forecast"
    TEST_SEGMENT_SPECT_DIR = "data/test_segment_spect"
    TEST_SEGMENT_LABEL_DIR = "data/test_segment_labels"

    # Step 1: Self-supervised pretraining
    print("Starting self-supervised pretraining...")
    pretrained_model = pretrain_msm(PRETRAIN_DIR, epochs=50)

    # Step 2: Fine-tune for spectrum forecasting
    print("\nFine-tuning for spectrum forecasting...")
    forecast_model = finetune_forecasting(pretrained_model, FORECAST_DIR, epochs=30)

    # Step 3: Fine-tune for segmentation
    print("\nFine-tuning for spectrogram segmentation...")
    segmentation_model = finetune_segmentation(
        pretrained_model,
        SEGMENT_SPECT_DIR,
        SEGMENT_LABEL_DIR,
        epochs=50
    )

    # Step 4: Evaluate models
    print("\nEvaluating forecasting model...")
    forecast_acc = evaluate_forecasting(forecast_model, TEST_FORECAST_DIR)

    print("\nEvaluating segmentation model...")
    seg_matrix = evaluate_segmentation(
        segmentation_model,
        TEST_SEGMENT_SPECT_DIR,
        TEST_SEGMENT_LABEL_DIR
    )

    print("\nPipeline completed successfully!")

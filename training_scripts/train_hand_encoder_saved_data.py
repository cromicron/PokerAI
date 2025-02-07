from train_hand_encoder import calc_losses, load_checkpoint
from encoders.hand_encoder import PokerHandEmbedding
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multiprocessing import Process, Queue

# Constants
PATH_DATA_TRAIN = "/mnt/e/pokerAI/encoder_data/train/large"
PATH_DATA_EVAL = "/mnt/e/pokerAI/encoder_data/eval"
MODEL_SAVE_PATH = "best_model.pth"
BEST_MODEL_PATH = "best_model.pth"
embedding_dim = 4
feature_dim = 256
deep_layer_dims = (512, 2048, 2048, 2048)
intermediary_dim = 16
batch_size = 32768
num_epochs = 100
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early Stopping & ReduceLROnPlateau
patience_lr = 4
patience_early_stop = 200

epochs_no_improve = 0

# **STEP 1: Initialize Model Without Loading Checkpoint**
model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# **STEP 2: Compute Initial Loss Magnitudes**
print("\n🔍 Computing initial loss magnitudes on untrained model...")

# Load evaluation dataset
eval_features = np.load(os.path.join(PATH_DATA_EVAL, "features_eval.npy"))
with open(os.path.join(PATH_DATA_EVAL, "labels_eval.pkl"), "rb") as f:
    eval_labels = pickle.load(f)

# Convert features to tensors
eval_features_tensor = torch.tensor(eval_features, dtype=torch.long, device=device)

# Split features
preflop_eval = eval_features_tensor[:, :2]
flop_eval = eval_features_tensor[:, 2:5]
turn_eval = eval_features_tensor[:, 5:6]
river_eval = eval_features_tensor[:, 6:7]

# Forward pass on untrained model
with torch.no_grad():
    _, eval_results = model(preflop_eval, flop_eval, turn_eval, river_eval)
    initial_losses = calc_losses(eval_results, eval_labels)  # REVERT TO PREVIOUS WORKING STATE

# Compute loss weights
loss_weights = {task: 1.0 / max(initial_losses[task].item(), 1e-6) for task in initial_losses}
print(f"✅ Initial loss weights computed: {loss_weights}\n")

# **STEP 3: Load Checkpoint & Define Scheduler**
load_checkpoint(MODEL_SAVE_PATH, model, optimizer)

model.freeze_except(["head_outcome_probs_river"])
with torch.no_grad():
    _, eval_results = model(preflop_eval, flop_eval, turn_eval, river_eval)
    eval_losses = calc_losses(eval_results, eval_labels)  # REVERT TO PREVIOUS WORKING STATE
    best_val_loss = eval_losses["outcome_river"]
    print(f"✅ Eval loss start computed: {best_val_loss}\n")


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=patience_lr,
                                                       verbose=True)

# Get all files and correctly pair feature-label files
files = sorted([f for f in os.listdir(PATH_DATA_TRAIN) if os.path.isfile(os.path.join(PATH_DATA_TRAIN, f))])
n_splits = len(files) // 2

# Remaining files are for training
feature_files = files[:n_splits]
label_files = files[n_splits:]

assert len(feature_files) == len(label_files), "Mismatch between training feature and label files!"


# Custom Dataset Class
class PokerDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = np.load(feature_path)
        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.labels.items()}

    # Preloading Data using Multiprocessing


def preload_data(feature_files, label_files, queue):
    """Loads data in a separate process to ensure GPU training doesn't get blocked."""
    while True:
        for feature_file, label_file in zip(feature_files, label_files):
            dataset = PokerDataset(os.path.join(PATH_DATA_TRAIN, feature_file), os.path.join(PATH_DATA_TRAIN, label_file))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            queue.put((feature_file, dataloader))  # Push file name and dataloader
        queue.put(None)  # End signal


# Start Data Preloading
data_queue = Queue(maxsize=2)
preload_process = Process(target=preload_data, args=(feature_files, label_files, data_queue))
preload_process.start()

# Training Loop
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch + 1}/{num_epochs} 🚀")
    model.train()

    total_train_loss = 0
    num_batches = 0
    outcome_river_loss_total = 0
    outcome_turn_loss_total = 0
    type_river_loss_total = 0

    tqdm_bar = tqdm(total=len(feature_files), desc=f"Epoch {epoch + 1}", unit="file", dynamic_ncols=True)

    while True:
        data = data_queue.get()
        if data is None:
            break

        file_name, dataloader = data
        for features, labels in dataloader:
            features = torch.tensor(features, dtype=torch.long, device=device)

            # Split features
            preflop, flop, turn, river = features[:, :2], features[:, 2:5], features[:, 5:6], features[:, 6:7]

            # Forward pass
            _, results = model(preflop, flop, turn, river)

            # Compute loss with scaling
            losses = calc_losses(results, labels)
            optimizer.zero_grad()
            total_loss = sum(loss_weights[task] * losses[task] for task in losses)

            # Extract required losses for logging
            outcome_river_loss = losses.get("outcome_river", torch.tensor(0.0, device=device)).item()
            outcome_turn_loss = losses.get("outcome_turn", torch.tensor(0.0, device=device)).item()
            type_river_loss = losses.get("type_river", torch.tensor(0.0, device=device)).item()

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Aggregate losses
            total_train_loss += total_loss.item()
            outcome_river_loss_total += outcome_river_loss
            outcome_turn_loss_total += outcome_turn_loss
            type_river_loss_total += type_river_loss
            num_batches += 1

            tqdm_bar.set_postfix(
                avg_loss=f"{total_train_loss / num_batches:.6f}",
                outcome_river=f"{outcome_river_loss_total / num_batches:.6f}",
                type_river=f"{type_river_loss_total / num_batches:.6f}",
                outcome_turn=f"{outcome_turn_loss_total / num_batches:.6f}",
                file=file_name,
            )

        tqdm_bar.update(1)

        # **Evaluate after each file**
        model.eval()
        with torch.no_grad():
            _, eval_results = model(preflop_eval, flop_eval, turn_eval, river_eval)
            eval_losses = calc_losses(eval_results, eval_labels)  # REVERTED BACK TO WORKING STATE
            total_eval_loss = sum(loss_weights[task] * eval_losses[task] for task in eval_losses if task != "outcome_turn").item()

        print(f"📊 Eval total: {total_eval_loss:.8f}, PR: {eval_losses['outcome_river']:.8f}, TR: {eval_losses['type_river']:.8f}, PT: {eval_losses['outcome_turn']:.8f}")

        # ReduceLROnPlateau after each file
        scheduler.step(eval_losses['outcome_river'])

        if eval_losses['outcome_river'] < best_val_loss:
            best_val_loss = eval_losses['outcome_river']
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                       BEST_MODEL_PATH)
            print("🏆 Best model saved!")

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} files.")

        if epochs_no_improve >= patience_early_stop:
            print(f"🚨 Early stopping triggered! No improvement for {patience_early_stop} files.")
            break

    tqdm_bar.close()
    if epochs_no_improve >= patience_early_stop:
        break

preload_process.join()
print("🎉 Training complete!")

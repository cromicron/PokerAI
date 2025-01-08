import random
from multiprocessing import Process, Queue, Lock, set_start_method, Event
from collections import deque
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import combinations
import numpy as np
from modules.mlp_prob import MLPProbModule
from PokerGame.HandComparatorLookup import compare_hands
from encoders.card_encoder import encode_holecards
from encoders.state_encoder import encode_board
from encoders.strength_encoder import encode_strength
from PokerGame.NLHoldem import Deck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Simulate poker hands and generate data
def simulate_runs(holecards, deck):
    remaining_deck = set(deck) - set(holecards)

    # simulate preflop


    hole_encoded = encode_holecards(holecards)
    board = random.sample(remaining_deck, 7)

    hero_hand = list(holecards) + board[:-2]
    villain_hand = board
    strengths = compare_hands([hero_hand, villain_hand])
    if strengths[0] == strengths[1]:
        result = 0
    elif strengths[0] > strengths[1]:
        result = 1
    else:
        result = 2


    board_encoded = encode_board(board)
    strength_preflop = encode_strength(holecards)
    strength_flop = encode_strength(holecards, board[:3])
    strength_turn = encode_strength(holecards, board[:4])
    strength_river = encode_strength(holecards, board)

    preflop_encoded = np.hstack([hole_encoded, np.zeros(70), strength_preflop])
    flop_encoded = np.hstack([hole_encoded, board_encoded[:31], np.zeros(39), strength_flop])
    turn_encoded = np.hstack([hole_encoded, board_encoded[:48], np.zeros(17), board_encoded[65:67], np.zeros(3), strength_turn])
    whole_hand_encoded = np.hstack([hole_encoded, board_encoded, strength_river])
    x = np.vstack([preflop_encoded, flop_encoded, turn_encoded, whole_hand_encoded])
    y = np.tile(result, (4,))
    return x, y

def producer(buffer, deck, chunk_size, buffer_filled_event):
    """
    Producer adds large shuffled chunks of hands to the buffer and periodically shuffles the entire buffer.
    """
    with tqdm(total=BUFFER_SIZE, desc="Filling Buffer", dynamic_ncols=True) as pbar:
        all_combinations = list(combinations(deck, 2))
        random.shuffle(all_combinations)

        local_buffer = []  # Temporary buffer for large chunks
        hands_added_since_shuffle = 0  # Track how many hands have been added since the last shuffle

        while True:
            for hole_cards in all_combinations:
                # Generate the 4 hands (preflop, flop, turn, river)
                x, y = simulate_runs(hole_cards, deck)
                samples = [(x_i, y_i) for x_i, y_i in zip(x, y)]  # Flatten hands into individual samples
                local_buffer.extend(samples)

                # When the local buffer reaches the chunk size, shuffle it and add to the global buffer
                if len(local_buffer) >= chunk_size:
                    random.shuffle(local_buffer)  # Shuffle the chunk locally

                    # Add shuffled chunk to the global buffer
                    for item in local_buffer:
                        buffer.put(item)  # Add items one by one to the Queue
                        hands_added_since_shuffle += 1
                    local_buffer.clear()  # Clear the local buffer after adding

                    # Signal that the buffer is fully filled for the first time
                    if buffer.qsize() >= BUFFER_SIZE and not buffer_filled_event.is_set():
                        buffer_filled_event.set()
                        print("Buffer is fully filled for the first time.")
                        pbar.close()

                    pbar.update(chunk_size)

                    # Shuffle the entire buffer after 1 million hands have been added
                    if hands_added_since_shuffle >= BUFFER_SIZE:
                        print("Shuffling entire buffer...")

                        # Extract all items, shuffle, and put them back
                        buffer_list = []
                        while not buffer.empty():
                            buffer_list.append(buffer.get())  # Remove all items from the queue
                        random.shuffle(buffer_list)  # Shuffle the entire buffer
                        for item in buffer_list:
                            buffer.put(item)  # Re-add shuffled items to the queue
                        hands_added_since_shuffle = 0  # Reset shuffle counter

            # Reshuffle all_combinations for continued randomness
            random.shuffle(all_combinations)



def consumer(queue, model, criterion, optimizer, batch_size, save_interval=1_000_000, save_path="model_checkpoint.pth"):
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    model.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    model.train()  # Set model to training mode
    total_samples = 0  # Total number of samples processed
    samples_since_last_save = 0  # Samples processed since the last checkpoint
    loss_window = deque(maxlen=100_000)  # Rolling window for loss tracking

    with tqdm(desc="Training Progress", total=None, dynamic_ncols=True) as pbar:
        while True:
            try:
                # Fetch a batch of data from the queue
                batch_data = []
                while len(batch_data) < batch_size:
                    item = queue.get()  # Blocking call to fetch data from the queue
                    batch_data.append(item)

                # Unpack features and labels
                x_s, y_s = zip(*batch_data)
                features = torch.tensor(np.vstack(x_s), dtype=torch.float32, device=device)
                labels = torch.tensor(np.hstack(y_s), dtype=torch.long, device=device)

                # Forward and backward passes with mixed precision
                with torch.cuda.amp.autocast():
                    logits = model(features)
                    loss = criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update metrics
                batch_size_actual = features.size(0)  # Account for smaller last batch
                total_samples += batch_size_actual
                samples_since_last_save += batch_size_actual
                loss_window.append(loss.item())
                running_avg_loss = sum(loss_window) / len(loss_window)

                # Update progress bar
                pbar.update(batch_size_actual)
                pbar.set_postfix({"Avg Loss": f"{running_avg_loss:.4f}"})

                # Save model checkpoint periodically
                if samples_since_last_save >= save_interval:
                    save_model(model, optimizer, total_samples, save_path)
                    samples_since_last_save = 0  # Reset save counter

            except Exception as e:
                print(f"Error in consumer: {e}")
                break



def save_model(model, optimizer, total_samples, path="model_checkpoint.pth"):
    torch.save({
        "total_samples": total_samples,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimizer=None, lr=None):
    """
    Load a checkpoint into the model and optimizer.
    """
    try:
        checkpoint = torch.load(path)  # Load the checkpoint file
        model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights

        if optimizer:  # If an optimizer is provided, load its state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Adjust learning rate if specified
            if lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        total_samples = checkpoint.get("total_samples", 0)  # Restore training progress
        print(f"Checkpoint loaded. Resumed from {total_samples} samples.")
        return total_samples

    except FileNotFoundError:
        print("No checkpoint found. Starting training from scratch.")
        return 0


BUFFER_SIZE = 1_000_00  # Buffer size for the multiprocessing queue
BATCH_SIZE = 512  # Batch size for training
SAVE_INTERVAL = 1_000_000  # Save model checkpoint every 1M samples
MODEL_SAVE_PATH = "model_checkpoint.pth"  # Path to save model checkpoints



if __name__ == "__main__":
    # Use 'spawn' to support CUDA with multiprocessing
    set_start_method("spawn", force=True)

    # Initialize deck
    deck = Deck()
    deck.shuffle()

    # Initialize model, loss function, and optimizer
    model = MLPProbModule(input_size=108, linear_layers=(1024, 512, 256, 128, 64))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)


    # Load checkpoint (optional)
    total_samples = load_checkpoint(MODEL_SAVE_PATH, model, optimizer)

    # Initialize multiprocessing queue and event
    remainder = BUFFER_SIZE % BATCH_SIZE
    true_buffer_size = BUFFER_SIZE + (BATCH_SIZE - remainder) if remainder != 0 else BUFFER_SIZE
    queue = Queue(maxsize=true_buffer_size)
    buffer_filled_event = Event()  # Event to signal buffer filling completion

    # Define and start the producer process
    producer_process = Process(target=producer, args=(queue, deck, BATCH_SIZE, buffer_filled_event))
    producer_process.daemon = True
    producer_process.start()
    print("Producer process started.")

    # Wait for the buffer to be fully filled
    print("Waiting for the buffer to fill completely...")
    buffer_filled_event.wait()  # Block until the producer signals completion

    # Define and start the consumer process
    consumer_process = Process(
        target=consumer,
        args=(queue, model, criterion, optimizer, BATCH_SIZE, SAVE_INTERVAL, MODEL_SAVE_PATH),
    )
    consumer_process.daemon = True
    consumer_process.start()
    print("Consumer process started.")

    # Manage processes and handle termination
    try:
        producer_process.join()
        consumer_process.join()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Terminating processes...")
        producer_process.terminate()
        consumer_process.terminate()
        producer_process.join()
        consumer_process.join()
        print("Processes terminated. Exiting.")

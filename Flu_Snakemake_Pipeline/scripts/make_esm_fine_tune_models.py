# code from ChatGPT
import torch
import esm
import argparse
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
import os


class ProteinDataset(Dataset):
    """Dataset class for protein sequences."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune ESM-2 on protein sequences.")
    parser.add_argument("--input", type=str, default="alignment.fasta", help="Input FASTA file containing training sequences.")
    parser.add_argument("--output", type=str, default="models/esm.bin", help="File path to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for optimizer.")
    parser.add_argument("--model", type=str, choices=["esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"],
                        default="esm2_t33_650M_UR50D", help="Specify which ESM-2 model to use.")
    return parser.parse_args()


def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences


def mask_tokens(tokens, mask_token_idx, vocab_size, device):
    """Apply masking to input tokens."""
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, 0.15, device=device)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time, replace masked tokens with the [MASK] token
    mask_indices = masked_indices & (torch.rand(labels.shape, device=device) < 0.8)
    tokens[mask_indices] = mask_token_idx

    # 10% of the time, replace masked tokens with random tokens
    random_indices = masked_indices & (torch.rand(labels.shape, device=device) < 0.1)
    random_tokens = torch.randint(0, vocab_size, labels.shape, device=device)
    tokens[random_indices] = random_tokens[random_indices]

    # The remaining 10% of the time, keep the original tokens
    return tokens, labels


def train_model(model, dataloader, batch_converter, optimizer, device, epochs, mask_token_idx, vocab_size, repr_layer):
    """Train the model on the given dataset."""
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for i, batch in enumerate(dataloader):

            # Convert batch of sequences using batch_converter
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)

            # Move batch tokens to the specified device
            batch_tokens = batch_tokens.to(device)

            # Mask tokens
            masked_tokens, labels = mask_tokens(batch_tokens, mask_token_idx, vocab_size, device)

            optimizer.zero_grad()
            results = model(masked_tokens, repr_layers=[repr_layer])
            logits = results["logits"]

            # Compute loss
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 1 == 0:
                print(f"Step {i + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")


def save_model(model, output_file):
    """Save the fine-tuned model to a file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(model.state_dict(), output_file)
    print(f"Model saved to {output_file}")


def main():
    args = parse_arguments()

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 650M parameter model: esm2_t33_650M_UR50D with embedding dimen of 1280
    # 3B parameter model: esm2_t36_3B_UR50D with embedding dimen of 2560
    # 15B parameter model: esm2_t48_15B_UR50D with embedding dimen of 6144

    # Map model names to their corresponding layer numbers
    model_layer_map = {
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
        "esm2_t48_15B_UR50D": 48
    }

    repr_layer = model_layer_map[args.model]

    # Load model and tokenizer
    print(f"Loading {args.model} model...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    batch_converter = alphabet.get_batch_converter()
    mask_token_idx = alphabet.mask_idx  # Index of the [MASK] token
    vocab_size = len(alphabet)  # Vocabulary size
    model = model.to(device)

    model.train()

    # Load training sequences
    print(f"Loading training sequences from {args.input}...")
    sequences = load_sequences(args.input)

    # Prepare dataset and dataloader
    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train model
    print("Starting fine-tuning...")
    train_model(model, dataloader, batch_converter, optimizer, device, args.epochs, mask_token_idx, vocab_size, repr_layer)

    # Save fine-tuned model
    print(f"Saving fine-tuned model to {args.output}...")
    save_model(model, args.output)


if __name__ == "__main__":
    main()
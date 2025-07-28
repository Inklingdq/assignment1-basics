import argparse
import time
import torch
from einops import rearrange

from cs336_basics.modules import AdamW, TransformerLM, cross_entropy, get_batch, gradient_clipping, load_checkpoint, lr_cosine_schedule, save_checkpoint
import numpy as np
import wandb


def train(args):
    if args.wandb_id:
       wandb.init(project="LMtransofrmer-Exp", id = args.wandb_id, resume ="allow")
    else: 
        wandb.init(project="LMtransofrmer-Exp", 
            config={
            "model": "GPT2-XL",
            "name": f"run-{args.experiment_name}-{time.strftime('%Y%m%d-%H%M%S')}",
            "context_length": args.context_length,
            "num_layers": args.num_layers,
            "vocab_size": args.vocab_size,
            "dataset": args.dataset_path
        })
    print(f"Training on {args.dataset_path}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Saving to {args.save_dir}, using device {args.device}")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        )
    model.to(args.device)    
    #model = torch.compile(model, backend="aot_eager")
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    data = np.memmap(args.dataset_path, dtype=np.uint16, mode='r')
    iter = 0
    if args.checkpoint_path:
        iter = load_checkpoint(args.checkpoint_path, model, optimizer)
    val_data = np.memmap(args.val_dataset_path, dtype=np.uint16, mode='r')
    start_time = time.time()
    for i in range(iter, args.total_steps):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        gradient_clipping(model.parameters(), args.gradient_clip)

        # Update lr
        current_lr = lr_cosine_schedule(i, args.l_min, args.lr, args.warmup_steps, args.cosine_annealing_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()

        if i % args.log_interval == 0:
            preds = logits.argmax(dim = -1)
            accuracy = (preds == y).float().mean().item()
            # After loss.backward()
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5  # L2 norm over all parameters
            print(f"Iter {i}, Training Loss: {loss.item():.4f}")
            wandb.log({
                "train/loss": loss.item(),
                "train/accuracy": accuracy,
                "grad_norm": grad_norm,
                "step": i,
                "time": time.time() - start_time
            }, step = i)
            if i % 500 == 0:
                save_checkpoint(model, optimizer, i, f"{args.save_dir}/model_iteration{i}.pt")
            # Validation loss
            model.eval()
            val_losses = []
            val_accuracy = []
            with torch.no_grad():
                for _ in range(10):
                    x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                    logits = model(x_val)
                    logits_flatten = rearrange(logits, "b c v -> (b c) v")
                    y_flatten = rearrange(y_val, "b c -> (b c)")
                    loss = cross_entropy(logits_flatten, y_flatten)
                    val_losses.append(loss.item())
                    preds = logits.argmax(dim = -1)
                    val_accuracy.append((preds == y_val).float().mean().item())
            model.train()
            print(f"Iter {i}, Validation Loss: {loss.item():.4f}")
            wandb.log({
                "val/loss": sum(val_losses)/len(val_losses),
                "val/accuracy": sum(val_accuracy)/len(val_accuracy),
                "step": i,
                "time": time.time() - start_time
            }, step = i)
            

 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument('--experiment-name', type = str, default = "test", help = "Name of experiment")
    parser.add_argument("--dataset-path", type=str, required = True, help = "Dataset file path for training")
    parser.add_argument("--val-dataset-path", type=str, required = True, help = "Dataset file path for validation")
    parser.add_argument("--checkpoint-path", type=str, default ="", help = "Checkpoint path to load from")
    parser.add_argument("--wandb_id", type=str, default ="", help = "Wandb id to continue logging")
    parser.add_argument("--batch-size", type=int, default = 32, help = "Batch Size")
    #parser.add_argument("--batch-size", type=int, default = 5, help = "Batch Size")
    parser.add_argument("--total-steps", type=int, default = 5000, help = "Dataset file path")
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--device', type = str, default = "mps", help = "Device to run inference on")

    # Add model hyperparameters
    parser.add_argument('--vocab_size', type=int, default = 10000)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    #parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=16)
    #parser.add_argument('--d-ff', type=int, default=512)
    parser.add_argument('--d-ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--context-length', type=int, default=256, help='Context length')
    # Optimization
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gradient-clip', type=float, default=1.0)

    # LR scheduler
    parser.add_argument('--l-min', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--warmup-steps', type=int, default=1000, help="Warmup steps before cosine decay")
    parser.add_argument('--cosine-annealing-steps', type=int, default=5000, help="Total steps for cosine decay")
    
    args = parser.parse_args()
    train(args)

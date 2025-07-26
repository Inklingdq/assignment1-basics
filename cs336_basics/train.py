import argparse
import time

from cs336_basics.modules import AdamW, TransformerLM, cross_entropy, get_batch, gradient_clipping, lr_cosine_schedule, save_checkpoint
import numpy as np
import wandb


def train(args):

    wandb.init(project="LMtransofrmer-Exp", config={
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
        theta=args.theta
            )    
    optimizer = AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    data = np.load(args.dataset_path, mmap_mode='r')
    start_time = time.time()
    for i, (x,y) in get_batch(data, args.batch_size, args.context_length):
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.gradient_clip)

        # Update lr
        current_lr = lr_cosine_schedule(i, args.l_min, args.l_max, args.warmup_steps, args.cosine_annealing_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()

        if i % args.log_interval == 0:
            print(f"Iter {i}, Loss: {loss.item():.4f}")
            wandb.log({
                "train.loss": loss.item(),
                "step": i,
                "time": time.time() - start_time
            }, step = i)
            save_checkpoint(model, optimizer, i, f"{args.save_dir}/model_iteration{i}.pt")
 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model.")
    parser.add_argument('--experiment-name', type = str, default = "test", help = "Name of experiment")
    parser.add_argument("--dataset-path", type=str, required = True, help = "Dataset file path")
    parser.add_argument("--batch-size", type=int, default = 5, help = "Batch Size")
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--device', type = str, default = "mps", help = "Device to run inference on")

    # Add model hyperparameters
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--context_length', type=str, default='32', help='Context length')
    # Optimization
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # LR scheduler
    parser.add_argument('--l_min', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--l_max', type=float, default=3e-5, help="Maximum learning rate (initial lr)")
    parser.add_argument('--warmup-steps', type=int, default=1000, help="Warmup steps before cosine decay")
    parser.add_argument('--cosine-annealing-steps', type=int, required=True, help="Total steps for cosine decay")
    
    args = parser.parse_args()
    train(args)

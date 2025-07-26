import argparse
import torch
from cs336_basics.tokenizer import tokenizer

from cs336_basics.modules import TransformerLM, top_p_sampling

def decode(args):
    tokenizer_instance = tokenizer.from_file(args.vocab_filepath, args.merges_filepath, args.special_tokens)
    tokens = tokenizer_instance.encode(args.prompt)
    input = torch.Tensor(tokens).unsqueeze(0).to(args.device)
    lm = TransformerLM(...)
    checkpoint = torch.load(args.checkpoint_path)
    lm = lm.load_state_dict(checkpoint['model'])
    lm.to(args.device)
    lm.eval()
    results = []
    for i in range(args.maximum_generate_tokens):
        output = lm(input)
        output = torch.softmax(output, -1, args.tempareture)s
        output = top_p_sampling(output, args.top_p_sampling_threshold)
        result = torch.multinomial(output, 1).item()
        results.append(result)
        input = torch.concat([input, torch.Tensor([[result]], device = args.device)], dim = 1)
        if input.shape[1] > lm.context_length:
            input = input[:, -lm.context_length]     
    return tokenizer_instance.decode(torch.Tensor(results))    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Decode...")
    parser.add_argument("--checkpoint-path", type=str, required = True, help = "Checkpoint path for model.")
    parser.add_argument("--user-prompt", type=str, required = True, help = "Please put in your prompt.")
    parser.add_argument("--maximum-generate-tokens", type = int, default = 100, help = "Maxium number of tokens to genrate.")
    parser.add_argument("--tempareture", type = float, default = 0.0, help = "Temperature for softmax.")
    parser.add_argument("--top-p-sampling-threshold", type = float, value = 1.0, help="Probability threshold for top p sampling.")
    parser.add_argument('--device', type = str, default = "mps", help = "Device to run inference on")

    # tokenizer filepath
    parser.add_argument("--vocab-filepath", type = str, required = True, help="Vacabulary file path.")
    parser.add_argument("--merges-filepath", type = float, value = 0.0, help="Tokenizer merge file path.")

    args = parser.parse_args()
    decode(args)
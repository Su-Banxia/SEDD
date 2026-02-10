import torch
import argparse
import os

from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling
import utils
import graph_lib
import noise_lib
from model import SEDD


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--prefix", type=str, default="Hi, my name is")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    prefix_ids = tokenizer(args.prefix).input_ids
    suffix_ids = tokenizer(args.suffix).input_ids
    input_ids = prefix_ids + suffix_ids
    input_locs = list(range(len(prefix_ids))) + list(range(args.length-len(suffix_ids), args.length))

    # more generaly commands can be defined with something like below:
    # input_ids = [0, 1, 512, 8080, 50256, 20000]
    # input_locs = [5, 6, 19, 20, 1000, 10001]


    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)

    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')
    
    if args.model_path.endswith('.pth'):
        exp_dir = os.path.dirname(os.path.dirname(args.model_path))
        print(f"Detected .pth file. Loading config from {exp_dir} and weights from {args.model_path}")
        
        # Manual loading to avoid load_model's hardcoded checkpoint path
        cfg = utils.load_hydra_config_from_run(exp_dir)
        graph = graph_lib.get_graph(cfg, device)
        noise = noise_lib.get_noise(cfg).to(device)
        model = SEDD(cfg).to(device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'ema' in checkpoint:
            print("Loading EMA weights...")
            # EMA state dict contains 'shadow_params', which is a list of tensors
            # We need to manually copy these tensors to the model parameters
            ema_state = checkpoint['ema']
            shadow_params = ema_state['shadow_params']
            
            # Filter model parameters that require grad (same logic as EMA init)
            model_params = [p for p in model.parameters() if p.requires_grad]
            
            if len(shadow_params) != len(model_params):
                raise ValueError(f"EMA parameter count mismatch: got {len(shadow_params)}, expected {len(model_params)}")
                
            for param, shadow in zip(model_params, shadow_params):
                param.data.copy_(shadow.data)
                
        elif 'model' in checkpoint:
            print("Loading model weights...")
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model, graph, noise = load_model(args.model_path, device)
    
    

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, args.length), 'analytic', args.steps, device=device, proj_fun=proj_fun
    )

    samples = proj_fun(sampling_fn(model))

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i)
        print("=================================================")

if __name__=="__main__":
    main()
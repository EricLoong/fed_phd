import torch
import torch.nn as nn
import torch_pruning as tp
from diffusers.models.attention import Attention
from diffusers.models.resnet import Upsample2D, Downsample2D

# Function to prune and adjust the attention layers
def prune_and_adjust_attention_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            heads = module.heads
            head_dim = module.to_q.weight.size(1) // heads
            input_dim = module.to_q.weight.shape[1]

            print(f"Pruning and adjusting Attention layer {name} with heads={heads}, head_dim={head_dim}")

            with torch.no_grad():
                # Prune the weights and biases
                module.to_q.weight = nn.Parameter(module.to_q.weight[:heads * head_dim, :input_dim])
                module.to_k.weight = nn.Parameter(module.to_k.weight[:heads * head_dim, :input_dim])
                module.to_v.weight = nn.Parameter(module.to_v.weight[:heads * head_dim, :input_dim])
                module.to_out[0].weight = nn.Parameter(module.to_out[0].weight[:, :heads * head_dim])

                if module.to_q.bias is not None:
                    module.to_q.bias = nn.Parameter(module.to_q.bias[:heads * head_dim])
                if module.to_k.bias is not None:
                    module.to_k.bias = nn.Parameter(module.to_k.bias[:heads * head_dim])
                if module.to_v.bias is not None:
                    module.to_v.bias = nn.Parameter(module.to_v.bias[:heads * head_dim])
                if module.to_out[0].bias is not None:
                    module.to_out[0].bias = nn.Parameter(module.to_out[0].bias[:input_dim])

            print(f"Adjusted {name}: to_q, to_k, to_v, and to_out pruned and updated.")

def adjust_static_layers(model):
    for m in model.modules():
        if isinstance(m, Downsample2D) or isinstance(m, Upsample2D):
            if m.conv.weight.shape[1] != m.channels:
                print(f"Correct layer shape mismatch: from  {m.channels} to {m.conv.weight.shape[1]}")
                m.channels = m.conv.weight.shape[1]


def group_norm_prune(args, model, logger):
    # Define the example inputs
    if args.dataset == 'cifar10':
        image_size = 32
    elif args.dataset == 'celeba':
        image_size = 64
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")
    model.to(args.device)
    example_inputs = {'sample': torch.randn(1, 3, image_size, image_size).to(args.device), 'timestep': torch.ones((1,)).to(args.device)}

    # Define the importance criterion
    imp = tp.importance.GroupNormImportance(p=2)

    # Count the original model operations and parameters
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    #logger.info(f'MACs: {base_macs / 1e9:.4f} G, #Params: {base_nparams / 1e6:.4f} M')

    # Define the pruner
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=args.pruning_ratio,
        channel_groups={},
        ignored_layers=[model.conv_out],
    )

    # Apply pruning
    for g in pruner.step(interactive=True):
        g.prune()

    prune_and_adjust_attention_layers(model)
    adjust_static_layers(model)

    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    model.to(args.device)
    logger.info(
        f"Prune Ratio is {args.pruning_ratio}. MACs: {base_macs / 1e9:.4f} G -> {macs / 1e9:.4f} G, #Params: {base_nparams / 1e6:.4f} M -> {nparams / 1e6:.4f} M")
    model.to('cpu')
    return model



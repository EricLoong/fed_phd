from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        # Initialize the optimizer with learning rate and weight decay
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Debug: Print the keys in server_controls and client_controls
        print(f"Server controls keys: {server_controls.keys()}")
        print(f"Client controls keys: {client_controls.keys()}")

        # Iterate over parameter groups and parameters
        for group in self.param_groups:
            for p, param_name in zip(group['params'], group['name']):
                # Skip non-trainable parameters (e.g., diffusion parts)
                if p.grad is None or not p.requires_grad:
                    continue

                # Retrieve the control variates by parameter name
                try:
                    c = server_controls[param_name]
                    ci = client_controls[param_name]
                except KeyError as e:
                    print(f"Error: {e} not found in controls")
                    raise

                # Check for shape alignment and apply control variate
                if p.shape != c.shape or p.shape != ci.shape:
                    raise RuntimeError(f"Shape mismatch: parameter shape {p.shape}, "
                                       f"server control shape {c.shape}, client control shape {ci.shape}")

                # Apply control variate and update parameter
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp * group['lr']

        return loss

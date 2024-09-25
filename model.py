import torch.nn as nn

class HyperRadialNeuralCelularAutomata(nn.Module):
    def __init__(self, batch_size,no_frame_samples, input_window_size, device):
        super(HyperRadialNeuralCelularAutomata, self).__init__()
        self.device = device
        self.no_frame_samples = no_frame_samples
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.in_scale = (1 + self.input_window_size * 2)

    def init_weights(self):
        if isinstance(self, nn.Linear):
            # torch.nn.init.xavier_uniform(self.weight)
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Conv1d):
            self.weight.data.normal_(mean=0.0, std=1.0)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Parameter):
            self.data.normal_(mean=0.0, std=1.0)

    def weight_reset(self: nn.Module):
        reset_parameters = getattr(self, "reset_parameters", None)
        if callable(reset_parameters):
            self.reset_parameters()
        # NOTE : Making sure that nn.conv2d and nn.linear will be reset
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def forward(self, x):
        return x
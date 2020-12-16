"""
    @author: jhuthmacher
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Primitive MLP for downstream classification
    """

    def __init__(self, in_dim, num_class, hidden_layers = [128]):
        """ Initialization of the downstream MLP. Hidden dimension 128.

            Paramters:
                in_dim: int
                    Number of input features. Has to match the representation dimension.
                num_class: int
                    Number of classes. This defines the output dim of the model.
        """
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.Tanh()]

        for hl in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[hl-1], hidden_layers[hl]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], num_class))

        self.layers = nn.Sequential(*layers)

        #### TO DEVICE #####
        self.layers = self.layers.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def forward(self, x: torch.Tensor):
        """ Forward function

            Parameters:
                x: torch.Tensor
                    Input of dimension (batch, emb_dim)
            Return:
                torch.Tensor: Dimension (batch, num_class)
        """
        
        x = x.type('torch.FloatTensor').to(self.device)  
        x = self.layers(x)

        return x
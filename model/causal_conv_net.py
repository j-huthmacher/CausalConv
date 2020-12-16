"""
    Implementation of a causal convolutional network.

    @paper: https://arxiv.org/pdf/1803.01271.pdf (Temporal Convolution Network)
    @author: j-huthmacher
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# pylint: disable=too-many-instance-attributes
class CausalConvNet(nn.Module):
    """ Implementation of ConvNet with causal convolutions.

    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, c_in: int, c_out: [int],
                 seq_length: int, model_cfg: dict = None,
                 activation = None, weights = None):
        """ Initialization of CausalConvNet

            Parameters:
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set default values
        self.kernel_size = 2
        self.stride = 1
        self.dropout = 0.5
        self.batch_norm = True
        self.test = False

        # Assign model config
        self.__dict__ = {**self.__dict__, **model_cfg}
        
        # Calculate dynamic attributes.
        self.initial_padding = (self.kernel_size - 1)

        self.layers = [nn.BatchNorm1d(c_in).float().to(self.device)]
        self.final_dim = None        


        # Stacking causal conv layers together!
        for i, curr_out in enumerate(c_out[:-1] if len(c_out) > 1 else c_out):
            dilation_size = 2 ** i

            causal_c_in =  (c_in if i == 0 else c_out[i - 1])
            causal_c_out = curr_out

            causal_cfg = {
                "kernel_size": self.kernel_size,
                "padding": self.initial_padding,
                "stride": self.stride,
                "dilation": dilation_size,
                "dropout": self.dropout,
                "test": self.test  # Testing purpose? I.e. ignore activation etc.
            }

            self.layers += [CausalLayer(causal_c_in, causal_c_out, causal_cfg,
                                        weights=weights, activation=activation)]

            if i == 0:
                self.final_dim = self.layers[-1].get_output_shape(seq_length)
            else:
                self.final_dim = self.layers[-1].get_output_shape(self.final_dim)
        
        self.conv_net = nn.Sequential(*self.layers).to(self.device)

        self.layers = [nn.Linear(self.final_dim * curr_out, c_out[-1])]

        #################
        # Linear Layers #
        #################
        if hasattr(self, 'linear_layers'):

            for i, dim in enumerate(self.linear_layers):
                if i == 0:
                    self.layers += [nn.Linear(self.final_dim * c_out[i - 1], dim), nn.LeakyReLU()]
                else:
                    self.layers += [nn.Linear(self.linear_layers[i-1], dim), nn.LeakyReLU()]

                if hasattr(self, 'batch_norm') and self.batch_norm:
                    self.layers += [nn.BatchNorm1d(dim)]
                    
        self.linear_net = nn.Sequential(*self.layers).to(self.device)


    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor):
        """ Forward pass of the model.

            Parameters:
                x: torch.Tensor
                    Input data to feed in the CausalNet.
                    Dimension: [batch_size, num_features, seq_length]
            Return:
                torch.Tensor: Result of the forward pass.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)


        x = x.type("torch.FloatTensor").to(self.device)

        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        
        if hasattr(self, 'linear_net'):
            x = self.linear_net(x)

        return x

    def __repr__(self):
        """ String representation.
        """
        representation = "CausalConvNet\n"
        representation += f"+-- Kernel Size: {self.kernel_size}\n"
        representation += f"+-- Sequence Length: {self.input_length}\n"
        representation += f"+-- Stride: {self.stride}\n"
        representation += f"+-- Dropout: {self.dropout}\n"
        representation += f"+-- Initial Padding: {self.initial_padding}\n"
        representation += f"+-- Causal Layers (causal_input -> output): {self.hidden_dim} -> {self.final_dim}\n"

        representation += f"+-- Final Linear Layers (input -> hidden_units -> output) {self.final_dim} -> {self.output_dim} -> {self.last_linear_out}\n"

        return representation

    @property
    def is_cuda(self):
        """ Returns if the model using cuda
        """
        return next(self.parameters()).is_cuda

    @property
    def paramters(self):
        return f"Parameters {sum(p.numel() for p in self.parameters())}"


class RemoveTrail(nn.Module):
    """ This neural network component removes the trail (caused by padding)
        after a convolutional operation.

        To remove the trail at the end is necessary, because the padding in
        this context is "only" used for shifting the data to the right and ensure
        by this a causal convolution. The trail at the end that is created
        is unnecessary.
    """

    def __init__(self, trail_size: int):
        """ Initialization of the RemoveTrail-component.

            Parameters:
                trail_size: int
                    Number positions (here features) that we cut of.
        """
        super().__init__()
        self.trail_size = trail_size

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor):
        """ The forward pass throught tthe RemoveTrail component

            Parameters:
                x: torch.Tensor
                    Tensor containing the input for the forward pass.
                    I.e. the input on that we want to cut of the trail.

            Return:
                torch.Tensor: The same as the input, but with cutted of trail.
        """
        return x[:, :, :-self.trail_size].contiguous()


# pylint: disable=too-many-instance-attributes
class CausalLayer(nn.Module):
    """ This class represents a causal convolutional layer.

        The layer consists of:
            Conv1D -> RemoveTrail -> ReLU -> Droput

    """

    def __init__(self, c_in: int, c_out: int, layer_cfg: dict, weights=None, activation=True):
        """ Initialization of the Causal Layer.

            Parameters:
        """
        super().__init__()

        # Default values
        self.dropout = 0.5
        self.kernel_size = 2
        self.stride = 1
        self.dilation = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Assign layer config
        self.__dict__ = {**self.__dict__, **layer_cfg}

        # Responsible for the causality in the convolution
        if isinstance(self.kernel_size, tuple):
            self.dila_padding = (self.kernel_size[1] - 1) * self.dilation
        else:
            self.dila_padding = (self.kernel_size - 1) * self.dilation

        # Causal convolution, because of adapted padding!
        self.conv1 = nn.Conv1d(c_in,
                               c_out,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.dila_padding,
                               dilation=self.dilation,
                               bias=False)

        # Remove trailling padding
        self.remove_trail = RemoveTrail(self.dila_padding)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.downsample = nn.MaxPool1d(self.kernel_size, stride=self.stride)
       
        if weights is not None:          
            self.conv1.weight = torch.nn.Parameter(weights)
        else:
            self.init_weights()
        
        self.to(self.device)

    def get_output_shape(self, input_size: int):
        """ Calculate the ouput shape for the causal layer.

            Important: Since we use a dilation factor and adapt in
                       each further layer the padding based on this
                       factor we have to use the inital padding for
                       calculating the output dim, because the
                       diliation and the increasing padding compensate
                       each other.

            output = ((input_size - kernel + (2 * padding)) // stride) + 1
            output -= padding
            output = math.floor(((output - kernel) // stride) + 1)

            Parameters:
                input_size: int
                    The input size of this layer (i.e. the output size
                    of the previous layer)

            Return:
                int: Dimension of the output of this layer.
        """
        # Output dim after convolutional operation

        out = math.floor((((input_size + (2 * self.dila_padding) - (self.dilation * (self.kernel_size - 1))) - 1) / self.stride) + 1)
        # conv_out_shape = ((input_size - self.kernel_size + (2 * self.padding)) // self.stride) + 1
        # Subtract the cutted of trail
        out -= self.dila_padding
        # Output dim after max pooling
        if self.downsample is not None and not self.test:
            out = math.floor(((out - self.kernel_size) / self.stride) + 1)
        return out

    def init_weights(self):
        """ Initialization of the weights.

            For the initialization we use a normal distribution with mean 0
            and std 0.01
        """
        self.conv1.weight.data.normal_(0, 0.01)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor):
        """ Forward pass for the causal convolutional layer.

            At the end of the forward pass we downsample the data using
            a specific layer like max pooling.

            Parameter:
                x: torch.Tensor
                    Input to the forward pass. Dimension (N, C_in, L_in)

            Return:
                torch.Tensor: returns the result of the causal layer.
        """

        x = x.type("torch.FloatTensor").to(self.device)

        x = self.remove_trail(self.conv1(x))

        if not self.test:
            x = self.dropout(self.relu(x))
            x = x if self.downsample is None else self.downsample(x)
            x = self.relu(x)

        return x 

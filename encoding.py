"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod

import torch
import numpy as np
from typing import Literal
from math import ceil, sqrt, pi
from scipy.stats import norm
import random



class AbstractEncoder(ABC):
    def pool(self, data:torch.Tensor, neurons_count:int, pooling:Literal['avg', 'max', 'random']='avg'):
        """
        perform an average pooling to reshape an image
        """
        pool_funcs = {'avg': torch.nn.AvgPool1d, 'max': torch.nn.MaxPool1d}
        if pooling in pool_funcs.keys():
            pooler = pool_funcs[pooling]
        elif pooling == 'random':
            # a lambda function with same signature as torch poolers
            # it returns a random sample of the data
            pooler = lambda kernel_size, stride: lambda data: data.squeeze()[random.sample(range(data.shape[1]), -kernel_size[0] + 1 + data.shape[1])].unsqueeze(0)

        data = data.flatten().type(torch.float32)
        n_old = len(data)
        n_new = neurons_count
        # Handles the case that new is bigger in some dims
        if n_new > n_old:
            data_new = torch.zeros((n_new,), dtype=torch.float32)
            data_new[sorted(random.sample(range(n_new), len(data)))] = data
            data = data_new
        else:
            data = data.unsqueeze(0)
            kernel  = (max(n_old - n_new + 1, 1),)
            pooler = pooler(kernel_size=kernel, stride=1,)
            data = pooler(data).squeeze()
        return data
    
    
    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.tensor:
        pass


class TTFSEncoder(AbstractEncoder):
    def __init__(
        self,
        time: int,
        neurons_count: int = None,
        dt: float = 1.0,
        device: str = "cpu",
        theta:float = 1.0,
        epsilon:float = 0.1,
    ) -> None:
        """
        Parameterss
        ---------
        `time` : int
            Length of encoded tensor.
        `dt` : float, Optional
            Simulation time step. The default is 1.0.
        `device` : str, Optional
            The device to do the computations. The default is "cpu".
        `neurons_count`: int
            The desired number of neurons.
            All the passed image first converts to this size using avg pooling, and then encodes
        `tetha`: int
            The tetha param
        """
        self.time = time
        self.device = device
        assert epsilon < 1
        self.epsilon = epsilon
        self.theta = theta
        self.neurons_count = neurons_count
        
    def __call__(self, data: torch.Tensor, pooling:Literal['avg', 'max', 'random']='avg') -> torch.tensor:
        """
        compute the threshold as P_th(t) = `tetha` * exp(-t/`tau`)
        `tau` would be detemined such that the minimum pixel would also fires
        
        Parameters:
        -----
        `data`: The data to process

        `tetha`: threshold constant

        `epsilon`: data would be normalized in range [`epsilon`, 1]

        Returns:
        -----
        a tensor of size (`time`, `<data shape>`)
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = self.pool(data, self.neurons_count, pooling)
        spikes = torch.zeros((self.time,) + data.shape, dtype=torch.bool)
        temp = (data - data.min()) / (data.max() - data.min()) # data in range [0, 1]
        temp = (temp * (1-self.epsilon)) + self.epsilon # data in range [`epsilon`, 1]
        tau = -self.time / np.log(self.epsilon / self.theta)
        for t in range(self.time):
            threshold = self.theta * np.exp(-(t+1)/tau)
            spikes[t, :] = temp >= threshold
            temp[temp >= threshold] = 0
        return spikes.type(torch.int8)



class PoissonEncoder(AbstractEncoder):
    def __init__(
        self,
        time: int,
        neurons_count: int = None,
        dt: float = 1.0,
        device: str = "cpu",
        
    ) -> None:
        """
        Parameterss
        ---------
        `time` : int
            Length of encoded tensor.
        `device` : str, Optional
            The device to do the computations. The default is "cpu".
        `neurons_count`: int
            The desired number of neurons.
            All the passed image first converts to this size using avg pooling, and then encodes
        
        """
        self.time = time
        self.device = device
        self.neurons_count = neurons_count
        
    def __call__(self, data: torch.Tensor, pooling:Literal['avg', 'max', 'random']='avg') -> torch.tensor:
        """
        compute the threshold as P_th(t) = `tetha` * exp(-t/`tau`)
        `tau` would be detemined such that the minimum pixel would also fires
        
        Parameters:
        -----
        `data`: The data to process
        
        Returns:
        -----
        a tensor of size (`time`, `<data shape>`)
        
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        data = self.pool(data, self.neurons_count, pooling)
        data = data / 255 # data in range [0, 1]
        size = data.numel()
        # rates for the Poisson dist
        rate = torch.zeros(size,)
        rate[data != 0] = 1 / data[data != 0]


        # Create Poisson distribution and sample inter-spike intervals
        # incrementing those who are 0 to avoid zero intervals
        dist = torch.distributions.Poisson(rate=rate)
        intervals = dist.sample(sample_shape=torch.Size([self.time + 1]))
        intervals[:, data != 0] += (intervals[:, data != 0] == 0).float()

        # Calculate spike times by cumulatively summing over time dimension.
        times = torch.cumsum(intervals, dim=0).long()
        times[times >= self.time + 1] = 0

        # Create tensor of spikes.
        spikes = torch.zeros(self.time + 1, size,).byte()
        spikes[times, torch.arange(size)] = 1
        spikes = spikes[1:]

        return spikes.view(self.time, size)

class PositionalEncoder(AbstractEncoder):
    def __init__(
        self,
        time: int,
        neurons_count: int = None,
        dt: float = 1.0,
        device: str = "cpu",
        pooling:Literal['avg', 'max']='avg',
        std:float = None,
        padding:float = None,
        
    ) -> None:
        """
        # Parameters
        ---------
        ## General Params

        `time` : int
            Length of encoded tensor.
        `dt` : float, Optional
            Simulation time step. The default is 1.0.
        `device` : str, Optional
            The device to do the computations. The default is "cpu".
        `neurons_count`: int
            The desired number of neurons.
            All the passed image first converts to this size using avg pooling, and then encodes
        `pooling`: str
            the pooling method

        ## Specific Params

        `std`: number
        `padding`: number
        
        """
        self.time = time
        self.dt = dt
        self.device = device
        self.neurons_count = neurons_count
        self.pooling = pooling

        std = 255 / neurons_count if std == None else std
        padding = std if padding == None else padding
        self.padding = padding
        d = (255 - 2 * padding) / (neurons_count - 1)
        self.d = d
        means = padding + torch.arange(neurons_count) * d
        self.max_prob_inv = std * sqrt(2 * pi) # to scale prob to desired time
        self.normals = [norm(loc=mean, scale=std) for mean in means]
    def cal_times(self, values:torch.Tensor):
        # calculate that for each value, at which time should each neuron spikes
        values_spikes = []
        for i in range(len(values)):
            item = torch.zeros(len(self.normals), dtype=torch.int32)
            point = int((values[i] - self.padding) / self.d)
            interval = (max(point-3, 0), min(point+3, len(item)))
            # print('point: ', point)
            # print('src:', item[interval[0]:interval[1]])
            # print('dst: ', torch.tensor([self.normals[t].pdf(values[i].item()) for t in range(interval[0], interval[1])]))
            item[interval[0]:interval[1]] = torch.tensor([self.normals[t].pdf(values[i].item()) for t in range(interval[0], interval[1])])  * self.time * self.max_prob_inv
            values_spikes.append(item.unsqueeze(0))
        

        return torch.concat(values_spikes)

        
        
    def __call__(self, data: torch.Tensor, pooling:Literal['avg', 'max', 'random']='avg') -> torch.tensor:
        """
        compute the threshold as P_th(t) = `tetha` * exp(-t/`tau`)
        `tau` would be detemined such that the minimum pixel would also fires
        
        Parameters:
        -----
        `data`: The data to process
        
        Returns:
        -----
        a tensor of size (`time`, `<neurons count>`)
        
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        # data = self.pool(data, self.neurons_count, self.pooling)
        data = data.flatten()
        times = self.cal_times(data)
        print('times shape: ', times.shape)
        # neurons_spikes = [set(times[:, i].numpy()).symmetric_difference({0}) for i in range(times.shape[1])]
        # spikes = torch.tensor([[] for t in range(self.time)])

        print(f"times: {(times != 0).count_nonzero()}")
        spikes = torch.zeros((self.time, self.neurons_count), dtype=torch.int8)
        for inp in times:
            # each row of input is for each data element
            for neuron in range(len(inp)):
                time = inp[neuron].item()
                if time:
                    spikes[time, neuron] = 1
        
        
        return spikes
        
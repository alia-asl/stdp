from pymonntorch import Behavior, NeuronGroup
import torch
from typing import Literal
import random

from encoding import *

class SetdtBehavior(Behavior):
  def initialize(self, net):
    net.dt = self.parameter("dt", 1.0)

# redefine __init__() for doc
class LIFBehavior(Behavior):
  DEFAULTS = {'base': {}, 'exp': {'delta': 2}}
  def __init__(self, func:Literal['base', 'exp']='exp', adaptive=True, Urest=-70, Uthresh=-50, Ureset=-75, Upeak=30.0, R=0.5, leak=True, tau_m=20, Tref=0, a=0.0, b=60, tau_w=30.0, variation=0.1, **func_kwargs):
    """
    # Parameter
    `func`: the linear or nonlinear function
    `adaptive`: whether it is adaptive or not
    -----
    ## Base LIF parameters
    `Urest`: number
    the resting potential
    `Uthresh`: number
    the Uthresh voltage which fires after that
    `Ureset`: number
    the voltage which the potential resets to that
    it would ignored for simple LIF
    `Upeak`: number
    the maximum voltage. After that, it comes back to `Ureset`
    it would set to `Uthresh` if `func`=`base`
    `R`: number
    the resistance. it is the inverse of g (in some texts)
    `leak`: bool
    whether to leak or not
    `tau_m`: number
    the parameter of LIF
    `Tref`: number
    the refractory time
    -----
    ## Adaption parameters
    `a`: number
    the subthreshold adaptation parameter
    `b`: number
    the spike-triggered adaptation parameter
    `tau_w`: number
    w parameter
    -----
    ## function parameters
    `func_kwargs`: the parameters of the function
    """
    super().__init__(func=func, adaptive=adaptive, Urest=Urest, Uthresh=Uthresh, Ureset=Ureset, Upeak=Upeak, R=R, leak=leak, tau_m=tau_m, Tref=Tref, a=a, b=b, tau_w=tau_w, variation=variation, func_kwargs=func_kwargs)
  def initialize(self, neurons:NeuronGroup):
    # getting the parameters
    self.func = self.parameter('func', None)
    self.adaptive = self.parameter('adaptive', None)
    neurons.Urest = neurons.vector(self.parameter('Urest', None))
    neurons.Uthresh = neurons.vector(self.parameter('Uthresh', None))
    neurons.Ureset = neurons.vector(self.parameter('Ureset', None))
    neurons.Upeak = neurons.vector(self.parameter('Upeak', None))
    variation = self.parameter('variation', None)
    for param in {'Urest', 'Uthresh', 'Ureset'}:
      old_val = getattr(neurons, param)
      dif =  (neurons.vector('random') - 0.5) * (old_val.abs() * variation)
      setattr(neurons, param, old_val + dif)
      
    if self.func in {'base'}:
      neurons.Upeak = neurons.Uthresh
    self.R = self.parameter('R', None)
    self.leak = self.parameter('leak', None)
    self.tau_m = self.parameter('tau_m', None)
    self.Tref = self.parameter('Tref', None)
    self.a = self.parameter('a', None)
    self.b = self.parameter('b', None)
    self.tau_w = self.parameter('tau_w', None)
    self.func_params = self.parameter('func_kwargs', None)
    self.dt = neurons.network.dt
    # init voltages
    neurons.voltage = neurons.Urest + neurons.vector(f"normal(0, {1 + variation * 5})")
    neurons.Tref = neurons.vector('zeros')
    if self.adaptive:
      neurons.w = neurons.vector('zeros')
    self.func_params_handle()

  def func_params_handle(self):
    for key in self.DEFAULTS[self.func].keys():
      setattr(self, key, self.func_params[key] if key in self.func_params.keys() else self.DEFAULTS[self.func][key])
      
      
  def forward(self, neurons:NeuronGroup):
    if not hasattr(neurons, 'inp'):
      raise AttributeError("An input behavior must set neurons input (`inp`) before calling LIF")
    firing = neurons.voltage >= neurons.Uthresh
    atPeak = neurons.voltage >= neurons.Upeak
    neurons.spike = atPeak.byte()
    neurons.voltage[atPeak] = neurons.Ureset[atPeak] # reset
    neurons.Tref[atPeak] = self.Tref
    if self.adaptive:
      neurons.w[atPeak] += (self.b * self.dt)

    # avoid decreasing voltage if threshold passed
    du = self.R * neurons.inp # get the input
    neurons.inp = neurons.vector(0) # reset input
    if self.leak:
      du[~firing] -= (neurons.voltage - neurons.Urest)[~firing] # leakage
      
    if self.func == 'exp':
      du += self.delta * torch.exp((neurons.voltage - neurons.Uthresh) / self.delta)
    if self.adaptive:
      dw = self.a * (neurons.voltage - neurons.Urest) - neurons.w
      dw = dw * self.dt / self.tau_w
      neurons.w += dw
      du[~firing] -= (self.R * neurons.w)[~firing]

    du = du * self.dt / self.tau_m
    neurons.voltage[neurons.Tref == 0] += du[neurons.Tref == 0] # U increases for those who are not spiked recently
    neurons.voltage = torch.clamp(neurons.voltage, max=neurons.Upeak, min=neurons.vector(-75)) # avoid super rapid growth
    # neurons.voltage[neurons.Tref > 0] -= du[neurons.Tref > 0] # and decrease for rest of them?
    neurons.Tref = torch.clamp(neurons.Tref - 1, min=0)
    
    
    

class InputBehavior(Behavior):
  def __init__(self, func=None, verbose=False, **func_args):
    """
    Parameters:
    -----
    `func`: the input behavior based on time.
    The input is the time and the size of neurons, and it should returns a vector of inputs

    `func_args`: you can pass your function args
    """
    super().__init__(func=func, verbose=verbose, func_args=func_args)
  def initialize(self, neurons:NeuronGroup):
    self.func = self.parameter('func', lambda t, d: neurons.vector(0))
    self.func_args = self.parameter('func_args', None)
    self.verbose:int = self.parameter('verbose', None)
    neurons.inp = neurons.vector(0)
    
    
  def forward(self, neurons:NeuronGroup):
    deltai = self.func(neurons.network.iteration, neurons.size, **self.func_args)
    neurons.inp += deltai

class ImageInput:
  def __init__(self, images:list, N:int, intersection:float, encoding:Literal['poisson', 'positional', 'TTFS']='poisson', time:int=10, sleep:int=10, amp:int=10, fix_image:bool=False) -> None:
    """
    Parameters
    -----
    `images`: list
      the list of images
    `N`: int
      the total number of neurons
    `intersection`: float
      the fraction of neurons that are used for all the images
    `encoding`: int
      the encoding method
    `time`: int
      the time of the image to show spikes
    `sleep`: int
      the duration of sleep
    """
    self.images = images
    encodings:dict[str, AbstractEncoder] = {'poisson': PoissonEncoder, 'positional': PositionalEncoder, 'TTFS': TTFSEncoder}
    n_inter = N * intersection # the number of intersected neurons
    n_sep = (N - n_inter) // len(images) # the number of separate neurons
    self.encoder:AbstractEncoder = encodings[encoding](neurons_count=n_inter + n_sep, time=time)
    self.fix_image = fix_image
    if fix_image:
      self.encodeds = [self.encoder(image) for image in images]
    self.N = N
    self.n_intersect = intersection
    self.n_sep = n_sep
    self.time = time
    self.sleep = sleep
    self.amp  = amp

    self.past_spikes:torch.tensor
    self.history:list[int] = [] # history of selected images
    
  def getImage(self, t, dim):
    """
    # Returns:
     a 1D tensor of size `n_inter` + `n_sep`
    """
    if t % (self.time + self.sleep) == 1:
      image_ind = random.choice(range(len(self.images))) # choose a random image
      self.history.append(image_ind)
      if self.fix_image:
        encoded_im = self.encodeds[image_ind]
      else:
        encoded_im = self.encoder(self.images[image_ind]) # a 2D tensor of shape ('time', 'n_inter' + 'n_sep')
      base = torch.zeros((self.time, self.N))
      # the first 'self.intersect' ones and the nth 'self.n_sep' would be set to encoded image
      base[:, :self.n_intersect] = encoded_im[:, :self.n_intersect]
      position = self.n_intersect + image_ind * self.n_sep
      base[:, position:position+self.n_sep] = encoded_im[:, self.n_intersect:]
      self.past_spikes = torch.concat((base, torch.zeros((self.sleep, self.N))))
    return self.past_spikes[(t-1) % (self.time + self.sleep), :] * self.amp





    
    
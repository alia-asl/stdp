import torch

def noise_input(t, dim, mean=0, std=1):
  """
  returns a noisy white noise
  `t` ignored and is just for integrity
  """
  return torch.normal(mean=mean, std=std, size=(dim,))

def step_input(t, dim, interval0=10, interval1=10, amp=20, noise=False, noise_mean=0, noise_std=1):
  """
  returns a periodic step input
  Parameters:
  -----
  `interval0`: int
  the interval of 0
  `interval1`: int
  the interval of 1
  `amp`: number
  the amplitude of spikes

  Returns:
  -----
  a spike input for `dim` neurons in the `t`'s second
  """
  if (t % (interval0 + interval1)) > interval0:
    ans = torch.ones(dim) * amp
  else:
    ans = torch.zeros(dim)
  if noise:
    ans += noise_input(t, dim, noise_mean, noise_std)
  return ans

def sin_input(t, dim, step=1/6, amp=20, noise=False, noise_mean=0, noise_std=1):
  """
  returns a periodic step input
  Parameters:
  -----
  `step`: float
  the steps of sin function
  `amp`: number
  the amplitude of spikes

  Returns:
  -----
  a spike input for `dim` neurons in the `t`'s second
  """
  ans = (torch.sin(torch.ones(dim) * torch.pi * t * step) + 1) * amp
  if noise:
    ans += noise_input(t, dim, noise_mean, noise_std)
  return ans


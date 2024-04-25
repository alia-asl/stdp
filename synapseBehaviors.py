from pymonntorch import Behavior, SynapseGroup, NeuronGroup
import torch
import random
from typing import Literal
from metrics import draw_weights

class DeltaBehavior(Behavior):
    DEFAULTS = {'density': {'fix_prob': 0.1, 'fix_count': 10}}
    def __init__(self, con_mode:Literal['full', 'fix_prob', 'fix_count']='full', 
                 density=None, w_mean=50, w_mu=5, rescale:bool=False, 
                 learn:Literal[False, 'stdp', 'rstdp']=False, flat=False, tau_pos=1, tau_neg=1, 
                 trace_dur=3, trace_amp=0.8, A_pos=1, A_neg=1, answer=None, reward=1,
                 save_changes_step=0):
        """
        # Parameters
        -----
        `con_mode`: `full`, `fix_prob`, or `fix_count`
            The connection type between the two Neuron Groups
        `density`: float in range (0, 1)
            The synapses density
            Value depends on `con_mode`:
            If =`full`, ignored
            If =`fix_prob`, it must be float (the prob)
            If =`fix_count`, it must be int (the number of pre-synaptic neurons)
        `rescale`: bool
            whether to rescale synaptic weights or not
        `learn`: bool
            whether to learn or not
        `flat`: bool
            whether to use flat method or not
        `tau_pos`: number
            pre-synaptic spike trace parameter
            ignored if `flat` is True
        `tau_neg`: number
            post-synaptic spike trace parameter
            ignored if `flat` is True
        `trace_dur`: number
            the duration of the trace
            ignored if `flat` is False
        `trace_amp`: number
            the amplitude of traces
            ignored if `flat` is False
        `A_pos` and `A_neg`: number
            parameters of the STDP
        `answer`: a callable
            At every step this function would be called
            to get the desired image index to know how to send reward
            This only would be used in RSTDP
        `reward`: number
            the amplitude of the reward function.
        """
        assert learn != 'rstdp' or answer != None
        if density == None:
            if con_mode == 'fix_prob':
                density = self.DEFAULTS['density']['fix_prob']
            elif con_mode == 'fix_count': 
                density = self.DEFAULTS['density']['fix_count']
            
        super().__init__(con_mode=con_mode, density=density, w_mean=w_mean, w_mu=w_mu, rescale=rescale, 
                         learn=learn, flat=flat, tau_pos=tau_pos, tau_neg=tau_neg, 
                         trace_dur=trace_dur, trace_amp=trace_amp, A_pos=A_pos, A_neg=A_neg, answer=answer,
                         reward=reward, save_changes_step=save_changes_step)
    def initialize(self, syn:SynapseGroup):
        self.init_W(syn)
        self.learn = self.parameter('learn', None)
        if self.learn in ['stdp', 'rstdp']:
            self.init_stdp(syn)
        else:
            self.parameter('flat', None)
            self.parameter('trace_dur', None)
            self.parameter('trace_amp', None)
            self.parameter('tau_pos', None)
            self.parameter('tau_neg', None)
            self.parameter('A_pos', None)
            self.parameter('A_neg', None)
            self.parameter('answer', None)
            self.parameter('reward', None)

        
        self.save_changes_step = self.parameter('save_changes_step', None)
        if self.save_changes_step:
            syn.W_history = []
        
    def init_W(self, syn:SynapseGroup):
        con_mode = self.parameter('con_mode', 'full')
        if con_mode in ['full', 'fix_prob', 'fix_count']:
            getattr(self, "init_" + con_mode)(syn=syn)
        else:
            raise ValueError(f'The connection mode {con_mode} is not defined')
        
    def init_full(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None) # dummy call, because of the warning
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        N = syn.src.size
        if rescale:
            syn.W = syn.matrix(f'normal(mean={w_mean / N}, std={w_mu / N})')
        else:
            syn.W = syn.matrix(f'normal(mean={w_mean}, std={w_mu})')
        
    
    def init_fix_prob(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None)
        if density > 1 or density < 0:
            raise ValueError("If `con_mode` is `fix_prob`, then it must be a probability")
        density  = self.parameter('density', None)
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        N = syn.src.size
        is_connected = syn.matrix('uniform') <= density
        syn.W = syn.matrix(0)
        if rescale:
            syn.W[is_connected] = syn.matrix(f'normal(mean={w_mean / (N * density)}, std={w_mu / N})')[is_connected]
        else:
            syn.W[is_connected] = syn.matrix(f'normal(mean={w_mean}, std={w_mu})')[is_connected]
        
        

    
    def init_fix_count(self, syn:SynapseGroup):
        rescale = self.parameter('rescale', None)
        density = self.parameter('density', None)
        density = int(density)
        w_mean = self.parameter('w_mean', None)
        w_mu   = self.parameter('w_mu', None)
        if density > syn.src.size:
            raise ValueError("If `con_mode` is `fix_count`, it must be less than the source neurons size")
        syn.W = syn.matrix(0)
        for i in range(syn.dst.size):
            pre_neurons = random.sample(range(syn.src.size), density)
            if rescale:
                syn.W[pre_neurons, i] = torch.normal(w_mean/density, w_mu/density, (len(pre_neurons), ))
            else:
                syn.W[pre_neurons, i] = torch.normal(w_mean, w_mu, (len(pre_neurons), ))

    def init_stdp(self, syn:SynapseGroup):
        syn.x = syn.src.vector(0)
        syn.y = syn.dst.vector(0)
        self.flat = self.parameter('flat', None)
        if self.flat: # for both cases the other's parameters called to bypass warnings
            self.trace_dur = self.parameter('trace_dur', None)
            self.trace_amp = self.parameter('trace_amp', None)
            self.parameter('tau_pos', None)
            self.parameter('tau_neg', None)

        else:    
            self.parameter('trace_dur', None)
            self.parameter('trace_amp', None)
            self.tau_pos = self.parameter('tau_pos', None)
            self.tau_neg = self.parameter('tau_neg', None)

        self.A_pos = self.parameter('A_pos', None)
        self.A_neg = self.parameter('A_neg', None)
        self.answer = self.parameter('answer', None)
        self.reward = self.parameter('reward', None)
        
    
    def update_stdp(self, syn:SynapseGroup):
        """
        update the variable x and y, and then W for STDP
        """
        oldw = syn.W.clone()

        if self.flat:
            syn.x += (self.trace_dur + 1) * syn.src.spike
            syn.y += (self.trace_dur + 1) * syn.dst.spike
            syn.x = torch.clamp(syn.x - 1, min=0)
            syn.y = torch.clamp(syn.y - 1, min=0)
            
        else:
            dx = -syn.x / self.tau_pos
            dx += syn.src.spike
            dx *= syn.network.dt
            syn.x += dx

            dy = -syn.y / self.tau_neg
            dy += syn.dst.spike
            dy *= syn.network.dt
            syn.y += dy

        if self.flat:
            post_pre = self.A_pos * (syn.src.spike.unsqueeze(1) * ((syn.y > 1) * self.trace_amp).unsqueeze(0)) # dim: (pre * 1) * (1 * post) = pre * post
            pre_post = self.A_neg * (((syn.x > 1) * self.trace_amp).unsqueeze(1) * syn.dst.spike.unsqueeze(0))
        
        else:
            post_pre = self.A_pos * (syn.src.spike.unsqueeze(1) * syn.y.unsqueeze(0)) # dim: (pre * 1) * (1 * post) = pre * post
            pre_post = self.A_neg * (syn.x.unsqueeze(1) * syn.dst.spike.unsqueeze(0))
        

        dw = pre_post - post_pre
        if self.learn == 'rstdp':
            dw += self.reward * self.get_reward(syn)
        dw = dw * syn.network.dt
        syn.W  += dw
        zeros_count = (dw == 0).sum(dim=0)
        zeros_count[zeros_count == 0] = dw.shape[0] # a tensor of size post, (number of zero dw for each post, n if no zero)
        zeros = dw == 0
        zeros[(~zeros.any(dim=0)).repeat(syn.src.size, 1)] = dw.shape[0]
        dw_reverse:torch.Tensor = zeros * (dw.sum(dim=0) / zeros_count) # reduce other weight for constant weights sum
        syn.W -= dw_reverse
        if torch.isnan(syn.W).any():
            print(f"dw: {dw}")
            print(f"dw^r = {dw_reverse}")
            raise ValueError("NaN found")
        
    def get_reward(self, syn:SynapseGroup) -> torch.Tensor:
        expect_spikes:torch.Tensor = self.get_pattern(syn)
        actual_spikes:torch.Tensor = syn.dst.spike
        reward = expect_spikes - actual_spikes
        return reward.repeat(20, 1)


    def get_pattern(self, syn:SynapseGroup) -> torch.Tensor:
        """
        returns the expected pattern
        a tensor of size post synaptic
        """
        spikes = syn.dst.vector(0)
        try:
            imInd = self.answer()
            spikes[imInd] = 1
        except:
            pass
        return spikes

    def forward(self, syn:SynapseGroup):
        if self.learn in ['stdp', 'rstdp']:
            self.update_stdp(syn)
        if self.save_changes_step and syn.network.iteration % self.save_changes_step == 0:
            syn.W_history.append(syn.W.clone())
        
        spikes = syn.src.spike.float()
        output = spikes @ syn.W
        if 'exc' in syn.tags:
            syn.dst.inp += output
        if 'inh' in syn.tags:
            syn.dst.inp -= output
            
class ConductanceBehavior(DeltaBehavior):
    DEFAULTS = {'density': {'fix_prob': 0.1, 'fix_count': 10}, 'g1': 5}
    def __init__(self, 
    con_mode:Literal['full', 'fix_prob', 'fix_count']='full', 
    density=None, 
    w_mean=50, w_mu=5, rescale=False,
    g0=0.0, g1 = 1, tau=10):
        """
        # Parameters
        ----
        `g0`: float\\
            the initial value of g
        `g1`: float\\
            It is used to calculate: Delta G = `g0` + `g1` * exp(-t/ `tau`)\\
            
        `tau`: float\\
            the time constant of conductance decay
        `density`: float in range (0, 1)\\
        the synapses density
        """
        if density == None:
            if con_mode == 'fix_prob':
                density = self.DEFAULTS['density']['fix_prob']
            elif con_mode == 'fix_count': 
                density = self.DEFAULTS['density']['fix_count']
        
        super().__init__(con_mode=con_mode, density=density, w_mean=w_mean, w_mu=w_mu, rescale=rescale, g0=g0, g1=g1, tau=tau)
    
    def initialize(self, syn:SynapseGroup):
        self.init_W(syn)
        # print("Synaptic weights:")
        # print(syn.W)
        self.g0 = self.parameter('g0', None)
        self.g1 = self.parameter('g1', None)
        syn.g = syn.src.vector(self.g0)
        self.last_spike_t = syn.src.vector(1000)
        self.alpha = self.parameter('alpha', None)
        self.tau = self.parameter('tau', None)
    
    def forward(self, syn:SynapseGroup):
        spikes = syn.src.spike
        self.last_spike_t[spikes.bool()] = 0
        spikes = spikes.float()
        syn.g = self.g0 + self.g1 * torch.exp(-self.last_spike_t/self.tau)
        output = (spikes * syn.g) @ syn.W
        if 'exc' in syn.tags:
            syn.dst.inp += output
        if 'inh' in syn.tags:
            syn.dst.inp -= output
        self.last_spike_t += 1
        

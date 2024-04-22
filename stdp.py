from pymonntorch import Network, NeuronGroup, SynapseGroup, Recorder, EventRecorder
from neuralBehaviors import *
from encoding import PoissonEncoder
from synapseBehaviors import DeltaBehavior

class STDP:
    def __init__(self, lif_params:dict={}, syn_params:dict={}, fix_image=False) -> None:
        """
        Params:
        -----
        `lif_params`: dict
            parameters of the LIF neurons
            keys can be:
            `func`, `adaptive`, `Urest`, `Uthresh`, `Ureset`, `Upeak`, `R`, `tau_m`, `Tref`, `a`, `b`, `tau_w`, `variation`, `func_kwargs`
        `syn_params`: dict
            parameters of the synapses
            keys can be:
            `con_mode`, `density`, `w_mean`, `w_mu`, `rescale`, `learn`, `flat`, `tau_pos`, `tau_neg`, `trace_dur`, `trace_amp`, `A_pos`, `A_neg`, `lr`
        
        """
        self.lif_params = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
        self.syn_params = {'tau_pos': 10, 'tau_neg': 10, 'learn': True, 'lr': 1, 'w_mean': 50, 'flat': True, }
        self.fix_image = fix_image

        for key in syn_params:
            self.syn_params[key] = syn_params[key]
        
        for key in lif_params:
            self.lif_params[key] = lif_params[key]
        
    
    def learn(self, image1, image2, intersection, image_dur=10, image_sleep=5, N=10, encoding:Literal['poisson', 'positinal', 'TTFS']='poisson', 
                iters=1000, inp_amp:int=10, verbose=False, ):
        """
        Parameters
        -----
        `imageX`: matrix
            the image to be encoded
        `image_dur`: int
            the duration of image spikes

        """
        
        # encoding images
        imInput = ImageInput(images=[image1, image2], N=N, intersection=intersection, encoding=encoding, time=image_dur, sleep=image_sleep, amp=inp_amp, fix_image=self.fix_image)
        self.image_dur = image_dur
        self.net = Network(behavior={1: SetdtBehavior()})
        self.ng_inp = NeuronGroup(N, behavior={
                1: LIFBehavior(**self.lif_params),
                2: InputBehavior(imInput.getImage, verbose=verbose),
                9: Recorder(variables=['inp']),
                10: EventRecorder(variables=['spike']),
            }, net=self.net, tag='pop_inp')

        self.ng_out = NeuronGroup(2, behavior={
                1: LIFBehavior(**self.lif_params),
                2: InputBehavior(),
                9: Recorder(variables=['inp']),
                10: EventRecorder(variables=['spike']),
            }, net=self.net, tag='pop_out')

        
        self.syn = SynapseGroup(net=self.net, src=self.ng_inp, dst=self.ng_out, behavior={3: DeltaBehavior(**self.syn_params)}, tag='exc')

        self.net.initialize(info=False)
        oldW = self.syn.W.clone()
        self.net.simulate_iterations(iters)
        self._train_spikes = self.ng_out['spike', 0]
        self._input_spikes = self.ng_inp['spike', 0]

        return {'images_history': imInput.history, 'oldW': oldW, 'newW': self.syn.W.clone()}
    def test(self) -> torch.Tensor:
        self.net.simulate_iterations(self.image_dur)
        return self.ng_out['spike', 0][self._train_spikes.shape[0]:]
        
    
    


from pymonntorch import Network, NeuronGroup, SynapseGroup, Recorder, EventRecorder
from neuralBehaviors import *
from synapseBehaviors import DeltaBehavior
from metrics import draw_weights
from matplotlib import pyplot as plt
from inputs import noise_input

class STDP:
    def __init__(self, N=10, lif_params:dict={}, syn_params:dict={}, fix_image=False) -> None:
        """
        Params:
        -----
        `N`: int
            the number of neurons
        `lif_params`: dict
            parameters of the LIF neurons
            keys can be:
            `func`, `adaptive`, `Urest`, `Uthresh`, `Ureset`, `Upeak`, `R`, `tau_m`, `Tref`, `a`, `b`, `tau_w`, `variation`, `func_kwargs`
        `syn_params`: dict
            parameters of the synapses
            keys can be:
            `con_mode`, `density`, `w_mean`, `w_mu`, `rescale`, `learn`, `flat`, `tau_pos`, `tau_neg`, `trace_dur`, `trace_amp`, `A_pos`, `A_neg`, `lr`
        
        """
        self.N = N
        self.lif_params = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
        self.syn_params = {'tau_pos': 10, 'tau_neg': 10, 'learn': 'stdp', 'w_mean': 50,}
        self.fix_image = fix_image

        for key in syn_params:
            self.syn_params[key] = syn_params[key]
        
        for key in lif_params:
            self.lif_params[key] = lif_params[key]
        
    
    def learn(self, image1, image2, intersection, image_dur=10, image_sleep=5, encoding:Literal['poisson', 'positinal', 'TTFS']='poisson', 
                iters=1000, inp_amp:int=10, verbose=False, W_changes_step=0):
        """
        Parameters
        -----
        `imageX`: matrix
            the image to be encoded
        `image_dur`: int
            the duration of image spikes

        """
        
        # encoding images
        imInput = ImageInput(images=[image1, image2], N=self.N, intersection=intersection, encoding=encoding, time=image_dur, sleep=image_sleep, amp=inp_amp, fix_image=self.fix_image)
        self.image_input = imInput
        self.image_dur = image_dur
        self.image_sleep = image_sleep
        self.net = Network(behavior={1: SetdtBehavior()})
        self.ng_inp = NeuronGroup(self.N, behavior={
                1: LIFBehavior(**self.lif_params),
                3: InputBehavior(noise_input),
                4: InputBehavior(imInput.getImage, verbose=verbose),
                9: Recorder(variables=['inp', 'voltage']),
                10: EventRecorder(variables=['spike']),
            }, net=self.net, tag='pop_inp')

        self.ng_out = NeuronGroup(2, behavior={
                1: LIFBehavior(**self.lif_params),
                2: InputBehavior(),
                9: Recorder(variables=['inp', 'voltage']),
                10: EventRecorder(variables=['spike']),
            }, net=self.net, tag='pop_out')

        
        self.syn = SynapseGroup(net=self.net, src=self.ng_inp, dst=self.ng_out, behavior={3: DeltaBehavior(**self.syn_params, save_changes_step=W_changes_step)}, tag='exc')

        self.net.initialize(info=False)
        oldW = self.syn.W.clone()
        self.net.simulate_iterations(iters)

        # Visualizing
        if W_changes_step:
            rows = ceil(len(self.syn.W_history) / 3)
            fig, ax = plt.subplots(rows, 3, figsize=(40, 10 * rows))
            fontsize=20
            for i in range(len(self.syn.W_history)):
                draw_weights(self.syn.W_history[i], ax[i // 3, i % 3])
                ax[i // 3, i % 3].set_title(f"iter {i * W_changes_step}", fontsize=fontsize)
            i += 1
            draw_weights(self.syn.W, ax[i // 3, i % 3])
            ax[i // 3, i % 3].set_title(f"The end", fontsize=fontsize)
            # fig.suptitle("How weights change", fontsize=40)
            # plt.show()
        self._train_spikes = self.ng_out['spike', 0].clone()
        self._input_spikes = self.ng_inp['spike', 0].clone()

        return {'images_history': imInput.history, 'oldW': oldW, 'newW': self.syn.W.clone()}
    def test(self) -> torch.Tensor:
        net = Network(behavior={1:SetdtBehavior()})
        ng_inp = NeuronGroup(self.N, behavior={
                1: LIFBehavior(**self.lif_params),
                2: InputBehavior(self.image_input.getImage,),
                9: Recorder(variables=['inp', 'voltage']),
                10: EventRecorder(variables=['spike']),
            }, net=net, tag='pop_inp')

        ng_out = NeuronGroup(2, behavior={
                1: LIFBehavior(**self.lif_params),
                2: InputBehavior(),
                9: Recorder(variables=['inp', 'voltage']),
                10: EventRecorder(variables=['spike']),
            }, net=net, tag='pop_out')

        
        syn = SynapseGroup(net=net, src=ng_inp, dst=ng_out, behavior={3: DeltaBehavior(**self.syn_params)}, tag='exc')

        net.initialize(info=False)
        syn.W = self.syn.W
        net.simulate_iterations(self.image_dur + self.image_sleep)
        
        return self.image_input.history[-1], ng_inp, ng_out
        
    
    


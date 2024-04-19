from pymonntorch import Network, NeuronGroup, SynapseGroup, Recorder, EventRecorder
from neuralBehaviors import *
from encoding import PoissonEncoder
from synapseBehaviors import DeltaBehavior


def learnSTDP(image1, image2, intersection, image_dur=1, N=10, encoding:Literal['poisson', 'positinal', 'TTFS']='poisson', 
              iters=1000, inp_amp:int=10, verbose=False, lif_params:dict={}, syn_params:dict={}):
    """
    Parameters
    -----
    `imageX`: matrix
        the image to be encoded
    `image_dur`: int
        the duration of image spikes

    `lif_params`: dict
        parameters of the LIF neurons
        keys can be:
        `func`, `adaptive`, `Urest`, `Uthresh`, `Ureset`, `Upeak`, `R`, `tau_m`, `Tref`, `a`, `b`, `tau_w`, `variation`, `func_kwargs`
    `syn_params`: dict
        parameters of the synapses
        keys can be:
        `con_mode`, `density`, `w_mean`, `w_mu`, `rescale`, `learn`, `flat`, `tau_pos`, `tau_neg`, `trace_dur`, `trace_amp`, `A_pos`, `A_neg`, `lr`
    """
    lif_params_default = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
    syn_params_default = {'tau_pos': 10, 'tau_neg': 10, 'learn': True, 'lr': 1, 'w_mean': 50, 'flat': True, }

    for key in syn_params:
        syn_params_default[key] = syn_params[key]
    
    for key in lif_params:
        lif_params_default[key] = lif_params[key]
    # encoding images
    imInput = ImageInput(images=[image1, image2], N=N, intersection=intersection, encoding=encoding, time=image_dur, amp=inp_amp)

    net = Network(behavior={1: SetdtBehavior()})
    ng_inp = NeuronGroup(N, behavior={
            1: LIFBehavior(**lif_params_default),
            2: InputBehavior(imInput.getImage, verbose=verbose),
            9: Recorder(variables=[]),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_inp')

    ng_out = NeuronGroup(2, behavior={
            1: LIFBehavior(**lif_params_default),
            2: InputBehavior(),
            9: Recorder(variables=[]),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_out')

    
    syn = SynapseGroup(net=net, src=ng_inp, dst=ng_out, behavior={3: DeltaBehavior(**syn_params_default)}, tag='exc')

    net.initialize(info=False)
    oldW = syn.W.clone()
    net.simulate_iterations(iters)

    return {'images_history': imInput.history, 'oldW': oldW, 'newW': syn.W.clone()}
    
    
    


from pymonntorch import Network, NeuronGroup, SynapseGroup, Recorder, EventRecorder
from neuralBehaviors import *
from encoding import PoissonEncoder
from synapseBehaviors import DeltaBehavior


def learnSTDP(image1, image2, intersection, image_dur=1, N=10, encoding:Literal['poisson', 'numeric', 'TTFS']='poisson', 
              lr:float=0.1, iters=1000, amp:int=10, w_mean:int=50):
    """
    Parameters
    -----
    `imageX`: matrix
        the image to be encoded
    `image_dur`: int
        the duration of image spikes
    """
    lif_params1 = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
    lif_params2 = {'tau_m': 10, 'a': -0.5, 'tau_w':  10, 'b':  20, 'R': 3, 'variation': 0.2}
    # encoding images
    imInput = ImageInput(images=[image1, image2], N=N, intersection=intersection, encoding=encoding, time=image_dur, amp=amp)

    net = Network(behavior={1: SetdtBehavior()})
    ng_inp = NeuronGroup(N, behavior={
            1: LIFBehavior(**lif_params1),
            2: InputBehavior(imInput.getImage,),
            9: Recorder(variables=['inp', 'voltage']),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_inp')

    ng_out = NeuronGroup(2, behavior={
            1: LIFBehavior(**lif_params2),
            2: InputBehavior(),
            9: Recorder(variables=['inp', 'voltage']),
            10: EventRecorder(variables=['spike']),
        }, net=net, tag='pop_out')

    
    syn = SynapseGroup(net=net, src=ng_inp, dst=ng_out, behavior={3: DeltaBehavior(tau_pos=10, tau_neg=10, learn=True, lr=lr, w_mean=w_mean)})

    net.initialize(info=False)
    oldW = syn.W.clone()
    net.simulate_iterations(iters)

    return {'images_history': imInput.history, 'oldW': oldW, 'newW': syn.W.clone()}
    
    
    


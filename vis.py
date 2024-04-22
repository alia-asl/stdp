import torch
import matplotlib.pyplot as plt

def draw_weights(weights:torch.Tensor, width=200, height=200, padding=10, scale_w=10, color='blue'):
    pre_x = torch.arange(padding, width - padding, (width - 2 * padding) // weights.shape[0])
    pre_y = torch.ones(weights.shape[0]) * padding
    post_x = torch.arange(padding, height - padding, (height - 2 * padding) // weights.shape[1])
    post_y = torch.ones(weights.shape[1]) * padding
    
    for i in range(len(pre_x)):
        for j in range(len(post_x)):
            plt.plot([pre_x[i], post_x[j]], linewidth=weights[i][j] / scale_w, color=color)
    
    plt.show()

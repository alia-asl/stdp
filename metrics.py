import torch
import numpy as np
import matplotlib.pyplot as plt

def draw_weights(weights:torch.Tensor, ax=None, width=40, height=20, padding=5, scale_w=10, color=(0.1, 0.6, 1), title=""):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        fig.suptitle(title, fontsize=40)
    pre_x = torch.arange(padding, height - padding + 0.001, (height - 2 * padding) / (weights.shape[0] - 1))
    post_x = torch.arange(padding, height - padding + 0.001 , (height - 2 * padding) / (weights.shape[1] - 1))
    
    color = np.array(color)
    for i in range(len(pre_x)):
        for j in range(len(post_x)):
            ax.plot([pre_x[i], post_x[j]], linewidth=(weights[i][j]) / scale_w, color=color if weights[i][j] > 0 else 1 - color)
            
def weights_similarity(weights:torch.Tensor):
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(weights[:, 0], weights[:, 1])
    return similarity
    
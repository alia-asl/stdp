import torch
# from wand.image import Image
# from wand.drawing import Drawing
# from wand.color import Color
import matplotlib.pyplot as plt

def draw_weights(weights:torch.Tensor, width=200, height=200, padding=10, scale_w=10):
    pre_x = torch.arange(padding, width - padding, (width - 2 * padding) // weights.shape[0])
    pre_y = torch.ones(weights.shape[0]) * padding
    post_x = torch.arange(padding, width - padding, (height - 2 * padding) // weights.shape[1])
    post_y = torch.ones(weights.shape[1]) * padding
    
    for i in range(len(pre_x)):
        for j in range(len(post_x)):
            plt.plot([pre_x[i], post_x[j]], linewidth=weights[i][j] / scale_w)
    # generate object for wand.drawing
    # with Drawing() as draw:
    #     # set stroke color
    #     draw.stroke_color = Color('green')
    #     for i in range(len(pre_x)):
    #         for j in range(len(post_x)):
    #             draw.stroke_width = weights[i][j]
    #             draw.line((pre_x[i], pre_y[i]), # Stating point
    #                     (post_x[j], post_y[j])) # Ending point
        
    #     with Image(width = width,
    #             height = height,
    #             background = Color('white')) as img:
    #         # draw shape on image using draw() function
    #         draw.draw(img)
    #     img.save(filename ='line.png')


import matplotlib.pyplot as plt
import torch


def plot_rect(image, label):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    width,height,dimension = image.shape
    # print(image.shape)
    # print("width : {} height : {}".format(width,height) )
    # width,height = 128,128
    tag = label['pred_logits']
    tag = torch.argmax(tag.view(-1, 9),1)
    print(tag)
    label = label['pred_boxes']
    index =0
    for coord in label:
        coord = coord.cpu()
        # print(coord.shape)
        index += 1
        for data in coord:
            x,y,w,h= data
            leftx = width*(x - w/2)
            lefty = height*(y - h/2)
            W = w * width
            H = h * height
            plt.gca().add_patch(
                plt.Rectangle(
                    xy=(leftx,lefty),
                    width=W,
                    height=H,
                    edgecolor='red',
                    fill=False, linewidth=1
                )
            )
            plt.text(leftx, lefty, '{}'.format(int(tag[index])), ha='center', va='center')
    return
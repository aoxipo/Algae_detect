import matplotlib.pyplot as plt

def plot_rect(image, label):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    width,height = image.shape
    for coord in label:
        class_number, x,y,w,h= coord 
        leftx = width*((x) - w/2)
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
        plt.text(leftx, lefty, '{}'.format(int(class_number)), ha='center', va='center')
    return 
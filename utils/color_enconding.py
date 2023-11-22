import matplotlib.pyplot as plt
import numpy as np
import colorsys
import matplotlib as mpl

cmaps = {}

hot = np.linspace(0.0, 1.0, 256)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
hot_gradient = np.vstack((gradient, hot))

def plot_color_bars():
    # Values for the HSV color encoding (0 to 180)
    hsv_values = np.arange(0, 181, 1)

    # Convert HSV values to RGB
    rgb_values_hsv = [colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0) for h in hsv_values]

    # Values for the hot color encoding (0.0 to 1.0)
    hot_values = np.linspace(0, 1, num=len(hsv_values))

    # Convert hot values to RGB using the 'hot' colormap
    cmap = plt.get_cmap('hot')
    rgb_values_hot = [cmap(h) for h in hot_values]

    # Create figure and axis
    fig, ax = plt.subplots(1, 2, figsize=(10, 1.2))

    # Plot the HSV color bar
    ax[0].bar(hsv_values, np.ones_like(hsv_values), color=rgb_values_hsv, width=1)
    ax[0].set_title('AoLP Color Encoding')
    ax[0].set_xlabel('values (0 - 180)')
    ax[0].set_yticks([])
    ax[0].set_xlim([0, 180])

    ax[1].imshow(hot_gradient, aspect='auto', cmap=mpl.colormaps['hot'])
    ax[1].set_title('DoLP Color Encoding')
    ax[1].set_xlabel('values (0.0 - 1.0)')
    ax[1].set_yticks([])
    # ax[1].set_xlim([0, 1])
    x_label_list = ['0.0', '0,25', '0.5', '0,75', '1.0']

    ax[1].set_xticks([0, 64, 128, 128+64, 255])

    ax[1].set_xticklabels(x_label_list)

    # Display the plot
    plt.tight_layout()
    path='./output/color_encoding.jpg'
    plt.savefig(path)


if __name__ == "__main__":
    plot_color_bars()


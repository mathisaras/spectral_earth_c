import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_rgb_pred_gedi(x, y_pred, labels, labels_masked, predictions_masked, false_color_ir=False):
    """Create a plot with three subpots:
        1) Image of EnMAP RGB patch
        2) Image of AGB predictions overlayed with GEDI shots
        3) Scatter plot with AGB labels vs predictions
    """

    rgb_indices = [43, 28, 10]
    false_color_infrared_indices = [69, 46, 28]

    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=3)

    # Select RGB bands and improve contrast
    if false_color_ir:
        image = x[false_color_infrared_indices]
    else:
        image = x[rgb_indices]
    image = image.numpy().transpose(1, 2, 0)

    image = (image - image.min()) / (image.max() - image.min())

    if false_color_ir:
        pass

    ax[0].set_title("Input RGB EnMAP patch")
    ax[0].imshow(image, interpolation='none')
    ax[0].axis("off")
    
    print(labels.shape)
    print("y_pred.shape: ", y_pred.shape)

    ax[1].set_title("Predicted AGB and GEDI shots")
    im1 = ax[1].imshow(y_pred.numpy().transpose(1, 2, 0), cmap='Greens', interpolation='none', vmin=0)
    gedi = ax[1].imshow(labels, interpolation='none')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical', label='Predicted AGB [Mg/ha]')
    ax[1].axis("off")

    ax[2].scatter(labels_masked, predictions_masked,
                  c=labels_masked, cmap='viridis')
    ax[2].axline((0, 0), slope=1, color='grey', ls="--")  # Add identity line
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gedi, cax=cax2, orientation='vertical', label='Reference AGB [Mg/ha]')
    ax[2].set_ylabel("Model prediction")
    ax[2].set_xlabel("Reference value (GEDI)")

    plt.tight_layout()
    return ax

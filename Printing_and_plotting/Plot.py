import matplotlib.pyplot as plt


def plt_histo(histo, show=True, save=True, file_name="histo"):
    plt.barh([str(i) for i in histo.keys()], histo.values(), color='g')
    plt.tick_params(labelsize=4)
    for i, v in enumerate(histo.values()):
        plt.text(v + .25, i - .25, str(v), color='blue', fontweight='normal', fontsize=6)
    if save:
        save_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/histogram/" + file_name
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.clf()


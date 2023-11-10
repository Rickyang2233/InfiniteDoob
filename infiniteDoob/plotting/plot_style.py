import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, cycler, axes
from tueplots.constants.color import palettes


def set_style():
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(figsizes.cvpr2022_half())
    plt.rcParams.update(fonts.neurips2021_tex(family="serif"))
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_muted))
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(axes.spines(top=False, right=False))




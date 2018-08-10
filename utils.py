from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def kde_scatter(x, y, colormap=None, size_factor=2, **kwargs):   
    if colormap is None:
        colormap = LinearSegmentedColormap.from_list(
        'sns_heat', [ sns.color_palette()[i] for i in [0, 3]], N=100)
    
    x = np.squeeze(x)
    y = np.squeeze(y)
    
    samples = np.vstack((x, y))
    
    samples = samples[:, np.all(~np.isnan(samples), axis=0)]
    
    densObj = kde( samples )

    def makeColours( vals ):
        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )

        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap=colormap).to_rgba( val ) for val in vals]

        return colours

    colours = makeColours( densObj.evaluate( samples ) )
    sizes   = densObj.evaluate( samples )
    sizes   = sizes / np.min(sizes) * size_factor

    return plt.scatter( samples[0], samples[1], color=colours, s=sizes, **kwargs)

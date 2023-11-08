## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.ticker as ticker

plt.rcParams['figure.figsize'] = 15,12
#plt.rcParams['figure.dpi'] = 200

from src.setup import * 
from src.utils import * 

############################
#various plotting functions#
############################

def newfig2d(nrows=1,ncols=1,plot_number=1,new_figure=True):
    if new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()
    plt.axis("equal")
    return (fig)
def newfig3d(nrows=1,ncols=1,plot_number=1,new_figure=True):
    if new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()
    ax = fig.add_subplot(nrows,ncols,plot_number,projection='3d')
    ax.set_xlim3d(-1,1), ax.set_ylim3d(-1,1), ax.set_zlim3d(-1,1)
    #ax.set_aspect("equal")
    return (fig,ax)
newfig = newfig3d # default

##### plot density estimation using embedding space metric
# adapted from http://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

def plot_density_estimate(M, obss, alpha=.2, limits=None, border=1.5, bandwidth=0.08, pts=100, cmap = cm.jet, colorbar=True):
    if obss.shape[1] > M.dim.eval():
        obss_q = np.array([M.get_coordsf(obs) for obs in obss])
    else:
        obss_q = obss
    kde = KernelDensity(bandwidth=bandwidth, metric='pyfunc', metric_params={"func":lambda q1,q2: np.linalg.norm((M.Ff(q1)-M.Ff(q2)))},
                    kernel='gaussian')
    kde.fit(obss_q)

    # grids
    obss_q_max = np.max(obss_q,axis=0)
    obss_q_min = np.min(obss_q,axis=0)
    minx = limits[0] if limits is not None else obss_q_min[0]-border
    maxx = limits[1] if limits is not None else obss_q_max[0]+border
    miny = limits[2] if limits is not None else obss_q_min[1]-border
    maxy = limits[3] if limits is not None else obss_q_max[1]+border
    X, Y = np.meshgrid(np.linspace(minx,maxx,pts),np.linspace(miny,maxy,pts))
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    xs = np.apply_along_axis(M.Ff,1,xy)
    X = xs[:,0].reshape(X.shape)
    Y = xs[:,1].reshape(X.shape)
    Z = xs[:,2].reshape(X.shape)

    # plot
    ax = plt.gca()
    fs = np.exp(kde.score_samples(xy))#/np.apply_along_axis(muM_Qf,1,xy)
    norm = mpl.colors.Normalize(vmin=0.,vmax=np.max(fs))
    colors = cmap(norm(fs)).reshape(X.shape+(4,))
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, facecolors = colors, linewidth=0., antialiased=True, alpha=alpha, edgecolor=(0,0,0,0), shade=False)
    m = cm.ScalarMappable(cmap=surf.cmap,norm=norm)
    m.set_array(colors)
    if colorbar:
        plt.colorbar(m, shrink=0.7)


#### plotting functions
# plot general function on M
def plot_f(M, f, F, minx, maxx, miny, maxy, alpha=.2, pts=100, cmap = cm.jet, vmin=None, vmax=None, colorbar=True):
        # grids        
        phi, theta = jnp.meshgrid(np.linspace(minx,maxx,pts),np.linspace(miny,maxy,pts))
        phitheta = jnp.vstack([phi.ravel(), theta.ravel()]).T
        xs = jax.vmap(F)(phitheta)
        X = xs[:,0].reshape(phi.shape)
        Y = xs[:,1].reshape(phi.shape)
        Z = xs[:,2].reshape(phi.shape)
        
        # plot
        ax = plt.gca()
        fs = jax.vmap(f,0)(xs)
        vector = len(fs.shape) > 1
        if not vector:
            if vmin is None or vmax is None:
                norm = mpl.colors.Normalize()
                norm.autoscale(fs)
            else:
                norm = mpl.colors.Normalize(vmin=vmin if vmin is not None else np.min(fs),vmax=vmax if vmax is not None else np.max(fs))
            colors = cmap(norm(fs)).reshape(phi.shape+(4,))
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, facecolors = colors, linewidth=0., antialiased=True, alpha=alpha, edgecolor=(0,0,0,0), shade=False)
            m = cm.ScalarMappable(cmap=surf.cmap,norm=norm)
            m.set_array(colors)
            if colorbar:
                plt.colorbar(m, ax=ax, shrink=0.7)
        else:
            M.plot()
            for i in range(fs.shape[0]):
                M.plotx(xs[i],v=fs[i])

def plot_sphere_f(M, f, alpha=.2, pts=100, cmap=cm.jet, vmin=None, vmax=None, colorbar=True, border = 1e-2):
    return plot_f(M, f, M.F_spherical, 0., 2.*np.pi, np.pi/2-border, -np.pi/2+border, alpha=alpha, pts=pts, cmap = cmap, vmin=vmin, vmax=vmax, colorbar=colorbar)

def plot_euclidean_f(M, f, minx, maxx, miny, maxy, pts=100, alpha=0.5, cmap=cm.jet, scale=50, vmin=None, vmax=None, colorbar=True, label=None):
    # grids
    assert M.__class__.__name__ == 'Euclidean', "M must be Euclidean manifold"
    if M.dim == 2:
        X, Y = jnp.meshgrid(np.linspace(minx, maxx, pts), 
                            np.linspace(miny, maxy, pts))
        coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    elif M.dim == 3:
        X, Y, Z = jnp.meshgrid(np.linspace(minx, maxx, pts), 
                               np.linspace(miny, maxy, pts),
                               np.linspace(miny, maxy, pts))
        coords = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    else:
        raise NotImplementedError("Only support 2D or 3D Euclidean manifold")
    
    # plot
    ax = plt.gca()
    fs = jax.vmap(f)(coords)
    vector = len(fs.shape) > 1
    if not vector:
        assert M.dim == 2, "Scalar visualization only support 2D Euclidean manifold"
        if vmin is None or vmax is None:
            norm = mpl.colors.Normalize()
            norm.autoscale(fs)
        else:
            norm = mpl.colors.Normalize(vmin=vmin if vmin is not None else fs.min(),
                                        vmax=vmax if vmax is not None else fs.max())
        colors = cmap(norm(fs)).reshape(X.shape + (M.dim+1, ))
        surf = ax.plot_surface(X, Y, fs.reshape(X.shape), facecolors=colors, cmap=cmap, linewidth=0., antialiased=True, alpha=alpha, edgecolor=(0, 0, 0, 0), shade=False)
        m = cm.ScalarMappable(cmap=surf.cmap, norm=norm)
        m.set_array(colors)
        if colorbar:
            plt.colorbar(m, ax=ax, shrink=0.7)
    else:
        M.plot()
        for i in range(fs.shape[0]):
            M.plotx(coords[i], u=fs[i], scale=scale)

# plot density estimate using spherical coordinates
def plot_sphere_density_estimate(M, obss_M, alpha=.2, bandwidth=0.08, pts=100, cmap = cm.jet):
        obss_M = np.apply_along_axis(lambda v: v/np.linalg.norm(v),1,obss_M)
        obss_q = np.apply_along_axis(M.F_spherical_invf,1,obss_M)        
        kde = KernelDensity(bandwidth=bandwidth, metric='pyfunc', metric_params={"func":lambda q1,q2: np.linalg.norm((M.F_sphericalf(q1)-M.F_sphericalf(q2)))},
                        kernel='gaussian')
        kde.fit(obss_q)
                            
        # grids
        phi, theta = np.meshgrid(np.linspace(0.,2.*np.pi,pts),np.linspace(-np.pi/2,np.pi/2,pts))
        phitheta = np.vstack([phi.ravel(), theta.ravel()]).T
        xs = np.apply_along_axis(M.F_spherical,1,phitheta)
        X = xs[:,0].reshape(phi.shape)
        Y = xs[:,1].reshape(phi.shape)
        Z = xs[:,2].reshape(phi.shape)
        
        # plot
        ax = plt.gca()
        fs = np.exp(kde.score_samples(phitheta))#/np.apply_along_axis(muM_Q_sphericalf,1,phitheta)
        norm = mpl.colors.Normalize(vmin=0.,vmax=np.max(fs))
        colors = cmap(norm(fs)).reshape(phi.shape+(4,))
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, facecolors = colors, linewidth=0., antialiased=True, alpha=alpha, edgecolor=(0,0,0,0), shade=False)
        m = cm.ScalarMappable(cmap=surf.cmap,norm=norm)    
        m.set_array(colors)
        plt.colorbar(m, shrink=0.7)


def plot_Euclidean_density_estimate(obss, alpha=.2, view='2D', limits=None, border=1.5, bandwidth=0.08, pts=100, cmap = cm.jet, colorbar=True):
        if view == '2D':
            if range is None:
                hist,histy,histx= np.histogram2d(obss[:,0],obss[:,1],bins=25)
            else:
                hist,histy,histx= np.histogram2d(obss[:,0],obss[:,1],bins=25,range=[[limits[0],          limits[1]],[limits[2],limits[3]]])
            extent = [histy[0],histy[-1],histx[0],histx[-1]]
            print(extent)

            #plt.contour(hist/np.max(hist),extent=extent,levels=[0.05,0.2,0.4,0.6],zorder=10)
            plt.imshow(hist.T/np.max(hist),extent=extent,interpolation='bicubic',         
                       origin='lower',cmap='Greys')#,levels=[0.05,0.2,0.4,0.6],zorder=10)
            #plt.colorbar()
        else:
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(obss)

            # grids
            obss_max = np.max(obss,axis=0)
            obss_min = np.min(obss,axis=0)
            minx = limits[0] if limits is not None else obss_min[0]-border
            maxx = limits[1] if limits is not None else obss_max[0]+border
            miny = limits[2] if limits is not None else obss_min[1]-border
            maxy = limits[3] if limits is not None else obss_max[1]+border
            X, Y = np.meshgrid(np.linspace(minx,maxx,pts),np.linspace(miny,maxy,pts))
            xy = np.vstack([X.ravel(), Y.ravel()]).T        

            # plot
            ax = plt.gca()
            fs = np.exp(kde.score_samples(xy))
            norm = mpl.colors.Normalize(vmin=0,vmax=np.max(fs))
            colors = cmap(norm(fs)).reshape(X.shape+(4,))
            surf = ax.plot_surface(X, Y, fs.reshape(X.shape), rstride=1, cstride=1, cmap=cmap, facecolors = colors,  linewidth=0., antialiased=True, alpha=alpha, edgecolor=(0,0,0,0), shade=False)
            m = cm.ScalarMappable(cmap=surf.cmap,norm=norm)
            m.set_array(colors)
            if colorbar:
                plt.colorbar(m, shrink=0.7)
            ax.set_xlim3d(minx,maxx), ax.set_ylim3d(miny,maxy), ax.set_zlim3d(0,np.max(fs))

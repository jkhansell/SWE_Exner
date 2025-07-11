# Plotting and debugging 
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

def plot_variables(obj):

    folder = "debug"
    fig, ax = plt.subplots(2,1,figsize=(12,6.5),sharex=True)
    ax[0].plot(obj.X[obj.X.shape[0]//2,:], obj.h[obj.h.shape[0]//2,:]+obj.z[obj.z.shape[0]//2,:],c="blue",
                marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
    ax[0].grid(alpha=0.25)
    ax[0].set_xlim(-2.75, 2.75)
    ax[0].set_ylim(1.18, 1.8)
    fig.suptitle("Time: {:.3f}".format(obj.etime))
    ax[0].set_ylabel("Free Surf level (m)")
    ax[1].plot(obj.X[obj.X.shape[0]//2,:], obj.z[obj.z.shape[0]//2,:],c="blue",
                marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
    ax[1].grid(alpha=0.25)
    ax[1].set_xlim(-2.75, 2.75)
    ax[1].set_ylim(0.9, 1.05)
    ax[1].set_ylabel("Bed level (m)")
    fig.suptitle("Time: {:.3f}".format(obj.etime))
    #plt.colorbar()
    fig.savefig(f"{folder}/debug_hz{i:06d}.png")
    plt.close()


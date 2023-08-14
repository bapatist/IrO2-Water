#%%
import numpy as np
from numpy import genfromtxt
import scipy as sp
import scipy.optimize
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 18}
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)
#%%
def main():
    fig, ax1 = plt.subplots()
    files  = ['1.txt', '2.txt', 'B.txt']
    colors = ['cornflowerblue', 'blue', 'midnightblue']
    for filename, clr_lbl in zip(files, colors):
        # Import data
        ct_1_data = genfromtxt(filename, delimiter=',')
        t_1, ct_1 = ct_1_data.T[0], ct_1_data.T[1]

        # Single Exponential Fit
        # K         = fit_single_exp(t_1, ct_1)
        # fit_1     = model_func_1(t_1, K)
        # Double Exponential Fit
        A, K1, K2 = fit_double_exp(t_1, ct_1)
        fit_2     = model_func_2(t_1, A, K1, K2)

        # plot(ax1, t=t_1, y_true=ct_1, fit_y=fit_1, 
        #      fit_parms=K, clr_lbl=clr_lbl)
        plot(ax1, t=t_1, y_true=ct_1, fit_y=fit_2, 
             fit_parms=(A, K1, K2), clr_lbl=clr_lbl)
        
    ax1.legend(bbox_to_anchor=(0.3,0.6), fancybox=True, shadow=True)
    plt.savefig("fit_O_bi_exp.png", dpi=300, bbox_inches='tight')

def model_func_1(t, K):
    return np.exp(-t/K)

def model_func_2(t, A, K1, K2):
    return A * np.exp(-t/K1) + (1-A) * np.exp(-t/K2)

def fit_single_exp(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func_1, t, y, maxfev=100)
    K = opt_parms
    return K

def fit_double_exp(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func_2, t, y, maxfev=1000)
    A, K1, K2 = opt_parms
    print(f'Time constant overall of bi exponential fit = ', f"{A*K1+(1-A)*K2:.2f}")
    return A, K1, K2

def plot(ax, t, y_true, fit_y, fit_parms, clr_lbl):
    if len(fit_parms)==3:
        A, K1, K2 = fit_parms
        ax.plot(t, fit_y, c=clr_lbl, ls='-', alpha=0.5,
                label='$C(t) = %0.3fe^{-t/%0.2f} + (1-%0.3f)e^{-t/%0.2f}$' % (A, K1, A, K2))
    elif len(fit_parms)==1:
        K = fit_parms
        ax.plot(t, fit_y, c=clr_lbl, ls='-', alpha=0.5, 
                label='$C(t) = e^{-t/%0.2f}$' % (K))  
    ax.plot(t, y_true, 'k.')
    ax.set(xlabel="$t,\ delay\ in\ ps$", ylabel="$C(t)$")
        
if __name__ == '__main__':
    main()
# %%

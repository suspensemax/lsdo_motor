import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.optimize import curve_fit

file_name = 'Magnetic alloy, silicon core iron C.tab'

''' --- FITTING FUNCTION FORMS --- '''
# x data is B, y-data is H
def fit_dep_B(x, a, b, c):
    f = (a * np.exp(b*x+c) + 200) * x**1.4
    return f

# x data is H, y-data is B
def fit_dep_H_old(x, coeff, order):
    f = 0;
    order = len(coeff) - 1
    for i in range(order + 1):
        f += coeff[i] * x**(order - i)
    return f

def fit_dep_H(x, a, b, c, d):
    # f = a * np.log(b*x+1)
    f = a * np.tanh(x/300 - .25) + .4
    # f = a * np.tanh(x/400)
    return f

''' --- FUNCTION THAT COMPUTES FITTINGS ---'''
def permeability_fitting(file_name=None, test=False, order=10):
    data = np.genfromtxt(file_name,skip_header = 1, delimiter = '\t')
    H_data = data[:,0]
    B_data = data[:,1]

    H_cont = np.linspace(0, H_data.max(), 15000)
    B_cont = np.linspace(0, B_data.max(), 15000)

    ## FITTING WITH X = H-DATA, Y = B-DATA ===> B = f(H)
    # fit_coeff_dep_H = np.polyfit(H_data, B_data, order)
    fit_coeff_dep_H, pconv = curve_fit(fit_dep_H, H_data[:15], B_data[:15])
    # print('Fitting coeff (dep. H): ', fit_coeff_dep_H)

    ## FITTING WITH X = B-DATA, Y = H-DATA ===> H = g(B)
    fit_coeff_dep_B, pconv = curve_fit(fit_dep_B, B_data, H_data)

    if test == False:
        return [fit_coeff_dep_H, fit_coeff_dep_B]
    elif test == True:
        data_dict = {
            'B_cont': B_cont,
            'H_cont': H_cont,
            'B_data': B_data,
            'H_data': H_data
        }
        return [fit_coeff_dep_H, fit_coeff_dep_B, data_dict]

if __name__ == '__main__':

    file_name = 'Magnetic alloy, silicon core iron C.tab'
    order=10
    
    mu_fitting = permeability_fitting(
        file_name=file_name,
        test=True,
        order=order
    )

    fit_coeff_dep_H = mu_fitting[0]
    fit_coeff_dep_B = mu_fitting[1]

    data_dict = mu_fitting[2]
    
    B_cont = data_dict['B_cont']
    H_cont = data_dict['H_cont']
    B_data = data_dict['B_data']
    H_data = data_dict['H_data']

    print(fit_coeff_dep_H)

    plt.figure(1)
    plt.plot(H_data, B_data, '*k', markersize=6, label='data')
    # plt.plot(H_cont, fit_dep_H_old(H_cont,fit_coeff_dep_H,order), label='fit')
    plt.plot(H_cont, fit_dep_H(H_cont, fit_coeff_dep_H[0], fit_coeff_dep_H[1], fit_coeff_dep_H[2], fit_coeff_dep_H[3]), label='fit')
    # plt.plot(H_cont, 1.55*np.tanh(H_cont/400 - .02), label='fit')
    plt.xlabel('H')
    plt.ylabel('B')
    plt.grid()
    plt.ylim(-0.1, 2)

    # plt.figure(2)
    # plt.plot(B_data, H_data, '*k', markersize=6, label='data')
    # plt.plot(B_cont, fit_dep_B(B_cont, fit_coeff_dep_B[0], fit_coeff_dep_B[1], fit_coeff_dep_B[2]),label='fit')
    # plt.xlabel('B')
    # plt.ylabel('H')
    # plt.grid()
    # plt.ylim(-200, 9000)
    # plt.legend()

    plt.show() 
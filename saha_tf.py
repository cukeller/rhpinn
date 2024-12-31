# solve Saha equation to determine degree of ionization of hydrogen in solar photosphere
# author: Christoph U.Keller, ckeller@nso.edu

# import math
import tensorflow as tf

def saha(T, rho):
    """
    T    Temperature in K
    rho  Density in kg/m^3
    """

    s = 5.53798845203817e-06 / rho * tf.pow(T, 1.5) * tf.exp(-157821.4464530811 / T)
    return (-s + tf.sqrt(tf.abs(s**2 + 4*s))) / 2

# Deep Neural Network for PINN
# author: Christoph U.Keller, ckeller@nso.edu

from scales import scales
import tensorflow as tf

def model():
    # neural network: input is (y,x,t,z), output is ux, uy, uz, log10(rho), log10(T)
    # all inputs and outputs are in physical units

    input_scale, input_offset, output_scale, output_offset = scales()
    ndim = len(input_scale)
    npar = len(output_scale)

    inputLayer = tf.keras.layers.Input(shape=(ndim,))
    inputScaling = tf.keras.layers.Rescaling(
            scale  = input_scale,
            offset = input_offset,
        )(inputLayer)
    dnn1 = tf.keras.layers.Dense( 32, activation='tanh')(inputScaling)
    dnn2 = tf.keras.layers.Dense( 64, activation='tanh')(dnn1)
    dnn3 = tf.keras.layers.Dense(128, activation='tanh')(dnn2)
    dnn4 = tf.keras.layers.Dense(128, activation='tanh')(dnn3)
    dnn5 = tf.keras.layers.Dense(128, activation='tanh')(dnn4)
    dnn6 = tf.keras.layers.Dense(npar, activation='linear')(dnn5)

    outputScaling = tf.keras.layers.Rescaling(
            scale  = output_scale,
            offset = output_offset
        )(dnn6)
    outputLayer = tf.keras.layers.concatenate([outputScaling, inputLayer])

    return tf.keras.Model(inputs=[inputLayer], outputs=[outputLayer])

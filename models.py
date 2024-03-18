import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

parallel_scan = tfp.math.scan_associative


# Dense layer with complex-valued parameters
class cDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, bias=False, **kwargs):
        super(cDense, self).__init__(**kwargs)
        self.kernel_imag = None
        self.kernel_real = None
        self.bias_imag = None
        self.bias_real = None
        self.units = units
        self.bias = bias
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        shape = input_shape

        self.kernel_real = self.add_weight("kernel_real",
                                           shape=[shape[-1], self.units],
                                           initializer="glorot_uniform",
                                           trainable=True)
        self.kernel_imag = self.add_weight("kernel_imag",
                                           shape=[shape[-1], self.units],
                                           initializer="glorot_uniform",
                                           trainable=True)
        self.bias_real = self.add_weight("bias_real",
                                         shape=[self.units],
                                         initializer="zeros",
                                         trainable=self.bias)
        self.bias_imag = self.add_weight("bias_imag",
                                         shape=[self.units],
                                         initializer="zeros",
                                         trainable=self.bias)

    @tf.function(experimental_compile=True)
    def call(self, inputs):
        real_input, imag_input = tf.math.real(inputs), tf.math.imag(inputs)

        real_output = tf.matmul(real_input, self.kernel_real) - tf.matmul(imag_input, self.kernel_imag) + self.bias_real
        imag_output = tf.matmul(real_input, self.kernel_imag) + tf.matmul(imag_input, self.kernel_real) + self.bias_imag

        if self.activation is not None:
            real_output = self.activation(real_output)
            imag_output = self.activation(imag_output)

        return tf.complex(real_output, imag_output)

    def get_config(self):
        config = super(cDense, self).get_config()
        config.update({"units": self.units, "activation": self.activation})
        return config


class complexMLP(tf.keras.Model):
    """
    Complex Multilayer Perceptron
    """

    def __init__(self, hidden_size, output_size, layer_num, name):
        super(complexMLP, self).__init__(name=name)
        self.layer_num = layer_num
        self.hidden_layers = [tf.keras.layers.Dense(hidden_size, activation='swish') for _ in range(layer_num)]

        # complex projection
        self.output_layer = cDense(output_size)

    @tf.function
    def call(self, inputs):
        for i in range(self.layer_num):
            inputs = self.hidden_layers[i](inputs)
        output = self.output_layer(tf.complex(inputs, tf.zeros_like(inputs)))
        return output


class complexNDM(tf.keras.Model):
    def __init__(self, hidden_size=32, output_size=4, layer_num=3,
                 sigma_min=0.9, sigma_max=0.999, phase=3.14, scan=True):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden_size should be even."
        self.hidden_size = hidden_size
        self.scan = scan  # Serial calculation or parallel scan

        u1 = np.random.random(size=int(hidden_size / 2))
        u2 = np.random.random(size=int(hidden_size / 2))

        # prior information
        v = -0.5 * np.log(u1 * (sigma_max ** 2 - sigma_min ** 2) + sigma_min ** 2)
        theta = u2 * phase

        # stable parameterization of the magnitude of the state matrix A
        self.v_log = tf.Variable(np.log(v), name='magnitude', dtype=tf.float32, trainable=True)

        # low oscillation frequency parameterization of the phase of the state matrix A
        self.theta_log = tf.Variable(np.log(theta), name='phase', dtype=tf.float32, trainable=True)

        # complex output matrix, which is complex-valued
        self.C = cDense(output_size, bias=False, name='C')

        # approximate the init hidden state
        self.f0 = complexMLP(hidden_size, hidden_size, layer_num, name='f_0')

        # map the control input into the hidden state space
        self.fu = complexMLP(hidden_size, hidden_size, layer_num, name='f_u')

    def effective_W(self):
        # exponential parameterization
        w = tf.math.exp(tf.complex(-tf.math.exp(self.v_log), tf.math.exp(self.theta_log)))
        
        # diagonal conjugate parameterization
        effective_w = tf.concat((w, tf.math.conj(w)), axis=0)
        return effective_w

    @tf.function(experimental_compile=True)
    def binary_operator_diag(self, element_i, element_j):
        # Binary operator for the parallel scan of the parallelizable neural dynamics model.
        # a_j * a_i is the power of state matrix A
        # a_j * u_i + u_j is the weighted prefix sum with the coefficient A
        a_i, u_i = element_i
        a_j, u_j = element_j
        return a_j * a_i, a_j * u_i + u_j

    @tf.function
    def call(self, inputs):
        """
        Args:
            inputs: [x0 (tensor): batch * seq * output_size, u (tensor): batch * n_steps * input_size]

        Returns:
            output (tensor): batch * n_steps * output_size
            hidden_states (tensor): (n_steps + 1) * batch * output_size
        """
        x0, u = inputs
        # init hidden state
        h0 = self.f0(x0)
        # the mapping of control input in the hidden state space
        ut = self.fu(u)
        # diagonal state matrix
        state_matrix = self.effective_W()

        # parallel scan
        if self.scan:
            # clone the state matrix as the shape of [batch, n_steps, state_matrix]
            state_matrix = tf.expand_dims(state_matrix, axis=0)
            state_matrix = tf.expand_dims(state_matrix, axis=0)
            state_matrix = tf.tile(state_matrix, [ut.shape[0], ut.shape[1], 1])

            elements = (tf.transpose(state_matrix, [1, 0, 2]), tf.transpose(ut, [1, 0, 2]))

            # power is [A, A^2, ... , A^n_steps] and inner_states is latter half component of Eq. (6)
            power, inner_states = parallel_scan(self.binary_operator_diag, elements, max_num_levels=10)

            # clone h0 as the shape of [n_steps, batch, h0]
            h = tf.expand_dims(h0, axis=0)
            h = tf.tile(h, [ut.shape[1], 1, 1])

            # return [A * h0, A^2 * h0, ... , A^n_steps * h0]
            h = tf.multiply(h, power)
            
            inner_states = tf.add(h, inner_states)
            outputs = tf.transpose(tf.math.real(self.C(inner_states)), [1, 0, 2])

            hidden_states = tf.concat([tf.expand_dims(h0, axis=0), inner_states], axis=0)
        # serial computing
        else:
            hidden_states = [h0]
            n_steps = ut.shape[1]
            for step in range(n_steps):
                h = tf.multiply(hidden_states[-1], state_matrix)
                h += ut[:, step, :]
                hidden_states.append(h)
            hidden_states = tf.stack(hidden_states)
            outputs = tf.transpose(tf.math.real(self.C(hidden_states[1:, :, :])), [1, 0, 2])

        return outputs, hidden_states

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

parallel_scan = jax.lax.associative_scan


class cDense(nn.Module):
    units: int
    bias: bool = False

    def setup(self):
        pass

    @nn.compact
    def __call__(self, inputs):
        input_size = inputs.shape[-1]

        kernel_real = self.param('kernel_real', nn.initializers.lecun_normal(), (input_size, self.units))
        kernel_imag = self.param('kernel_imag', nn.initializers.lecun_normal(), (input_size, self.units))

        if self.bias:
            bias_real = self.param('bias_real', nn.initializers.zeros, (self.units,))
            bias_imag = self.param('bias_imag', nn.initializers.zeros, (self.units,))

        real_input, imag_input = jnp.real(inputs), jnp.imag(inputs)

        real_output = jnp.dot(real_input, kernel_real) - jnp.dot(imag_input, kernel_imag)
        imag_output = jnp.dot(real_input, kernel_imag) + jnp.dot(imag_input, kernel_real)

        if self.bias:
            real_output += bias_real
            imag_output += bias_imag

        return real_output + 1j * imag_output


class complexMLP(nn.Module):
    hidden_size: int
    output_size: int
    layer_num: int

    @nn.compact
    def __call__(self, inputs):
        for _ in range(self.layer_num):
            inputs = nn.swish(nn.Dense(self.hidden_size)(inputs))
        inputs = inputs + 1j * jnp.zeros_like(inputs)
        output = cDense(self.output_size, bias=True)(inputs)
        return output


class complexNDM(nn.Module):
    hidden_size: int
    output_size: int
    layer_num: int
    sigma_min: float
    sigma_max: float
    phase: float
    scan: bool

    def setup(self):
        u1 = jnp.array(np.random.uniform(size=(int(self.hidden_size / 2),)))
        u2 = jnp.array(np.random.uniform(size=(int(self.hidden_size / 2),)))

        v = -0.5 * jnp.log(u1 * (self.sigma_max ** 2 - self.sigma_min ** 2) + self.sigma_min ** 2)
        theta = u2 * self.phase

        self.v_log = self.param('v_log', lambda rng, shape: jnp.log(v), ())
        self.theta_log = self.param('theta_log', lambda rng, shape: jnp.log(theta), ())

        self.C = cDense(self.output_size)
        self.f0 = complexMLP(self.hidden_size, self.hidden_size, self.layer_num)
        self.fu = complexMLP(self.hidden_size, self.hidden_size, self.layer_num)

    def effective_W(self):
        w = jnp.exp(-jnp.exp(self.v_log) + 1j * jnp.exp(self.theta_log))
        effective_w = jnp.concatenate((w, jnp.conj(w)), axis=0)
        return effective_w

    def __call__(self, inps):
        x0, u = inps
        h0 = self.f0(x0)
        ut = self.fu(u)
        state_matrix = self.effective_W()

        if self.scan:
            state_matrix = jnp.expand_dims(state_matrix, axis=0)
            state_matrix = jnp.expand_dims(state_matrix, axis=0)
            state_matrix = jnp.tile(state_matrix, (ut.shape[0], ut.shape[1], 1))

            elements = (jnp.transpose(state_matrix, (1, 0, 2)), jnp.transpose(ut, (1, 0, 2)))

            def binary_operator_diag(element_i, element_j):
                a_i, u_i = element_i
                a_j, u_j = element_j
                return a_j * a_i, a_j * u_i + u_j

            power, inner_states = parallel_scan(binary_operator_diag, elements)

            h = jnp.expand_dims(h0, axis=0)
            h = jnp.tile(h, (ut.shape[1], 1, 1))

            h = h * power
            inner_states = h + inner_states
            outputs = jnp.transpose(jnp.real(self.C(inner_states)), (1, 0, 2))

            hidden_states = jnp.concatenate([jnp.expand_dims(h0, axis=0), inner_states], axis=0)
        else:
            hidden_states = [h0]
            n_steps = ut.shape[1]
            for step in range(n_steps):
                h = hidden_states[-1] * state_matrix
                h += ut[:, step, :]
                hidden_states.append(h)
            hidden_states = jnp.stack(hidden_states)
            outputs = jnp.transpose(jnp.real(self.C(hidden_states[1:, :, :])), (1, 0, 2))

        return outputs, hidden_states

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexNN.nn import cLinear

class complexMLP(nn.Module):
    """
    Complex Multilayer Perceptron
    """
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                output_size: int, 
                layer_num: int
                ):
        super(complexMLP, self).__init__()
        self.layer_num = layer_num
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layer_num - 1)])

        # complex projection
        self.output_layer = cLinear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        x = F.gelu(self.input_layer(x))
        for i in range(self.layer_num - 1):
            x = F.gelu(self.hidden_layers[i](x))
        output = self.output_layer(torch.complex(x, torch.zeros_like(x)))
        return output


class complexNDM(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 seq_len: int, 
                 hidden_size: int, 
                 output_size: int, 
                 layer_num: int,
                 enable_mp: bool = False,
                 sigma_min: float = 0.9, 
                 sigma_max: float = 0.999, 
                 phase: float = np.pi / 10):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden_size should be even."
        self.hidden_size = hidden_size
        self._enable_mp = enable_mp

        u1 = np.random.random(size=int(self.hidden_size / 2))
        u2 = np.random.random(size=int(self.hidden_size / 2))

        # prior information
        v = -0.5 * np.log(u1 * (sigma_max ** 2 - sigma_min ** 2) + sigma_min ** 2)
        theta = u2 * phase

        self.v_log = nn.Parameter(torch.tensor(np.log(v), dtype=torch.float32))
        self.theta_log = nn.Parameter(torch.tensor(np.log(theta), dtype=torch.float32))

        # complex output matrix
        self.C = cLinear(hidden_size, output_size, bias=False)

        self.f0 = complexMLP(output_size * seq_len, hidden_size, hidden_size, layer_num)
        self.fu = complexMLP(input_size, hidden_size, hidden_size, layer_num)

    def _update_effective_w(self):
        w = torch.exp(-torch.exp(self.v_log) + 1j * torch.exp(self.theta_log))
        effective_w = torch.cat((w, w.conj()), dim=0)
        self._w_effect = torch.diag(effective_w)

    def forward(self, x0, u, h_step = 0):
        """
        Execute the following gray-box formula:

        h{0} = f_0(x0)
        h{t+1} = f_x(h{t}) + f_u(u{t})
        y{t} = f_y(h{t})

        Args:
            x0 (tensor): batch * seq * output_size
            u (tensor): batch * n_steps * input_size

        Returns:
            output (tensor): n_steps * batch * output_size
            hidden_states (tensor): n_steps * batch * output_size
        """
        if not self._enable_mp: 
          outputs, hidden_states = self._forward_sp(x0, u)
        elif self._enable_mp:
          outputs, hidden_states = self._forward_mp(x0, u, h_step)
      
        return outputs, hidden_states

    def _forward_mp(self, x0, u, h_step):
        batch_size, seq_len, input_size = u.size()
        u = u.reshape(-1, input_size)
        ut = self.fu(u)
        ut = ut.reshape(batch_size, seq_len, self.hidden_size)
        h0 = self.f0(x0)

        self._update_effective_w()

        hidden_state = self._forward_mp_impl(h_step, ut, h0)
        hidden_state = hidden_state.unsqueeze(0)

        outputs = self.C(hidden_state).real
        return outputs, hidden_state
    
    def _forward_sp(self, x0, u):
        batch_size, seq_len, input_size = u.size()
        u = u.reshape(-1, input_size)
        ut = self.fu(u)
        ut = ut.reshape(batch_size, seq_len, self.hidden_size)
        n_steps = ut.shape[1]

        h0 = self.f0(x0)
        hidden_states = [h0]

        self._update_effective_w()

        for step in range(n_steps):
            h = torch.matmul(hidden_states[-1], self._w_effect)
            h += ut[:, step, :]
            hidden_states.append(h)

        hidden_states = torch.stack(hidden_states)
        outputs = self.C(hidden_states[1:, :, :]).real
        return outputs, hidden_states

    def _forward_mp_impl(self, num_step, ut, h0):
        sum_u = torch.zeros_like(ut[:, 0, :], requires_grad=True) + ut[:, 0, :]
        for u_step in range(num_step):
            sum_u =  torch.matmul(sum_u, self._w_effect) + ut[:, u_step + 1, :]
        hidden_state = \
            torch.matmul(h0, 
                        torch.matrix_power(self._w_effect, num_step + 1)) + sum_u
        return hidden_state
  
    @property
    def w_effect(self):
        self._update_effective_w()
        return self._w_effect
             
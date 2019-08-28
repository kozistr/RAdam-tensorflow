"""AdaBound for Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re


class RAdamOptimizer(tf.train.Optimizer):
    """Optimizer that implements the RAdam algorithm.

    See [Liu et al., 2019](https://arxiv.org/abs/1908.03265)
    ([pdf](https://arxiv.org/abs/1908.03265)).
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 decay: float = 0.,
                 weight_decay: float = 0.,
                 exclude_from_weight_decay: list = None,
                 use_locking: bool = False,
                 name: str = "RAdam"):
        super(RAdamOptimizer, self).__init__(use_locking, name)

        if not 0. <= beta1 < 1.:
            raise ValueError("Invalid beta1 value : {}".format(beta1))
        if not 0. <= beta2 < 1.:
            raise ValueError("Invalid beta2 value : {}".format(beta2))
        if epsilon <= 0.:
            raise ValueError("Invalid epsilon value : {}".format(epsilon))

        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._decay = decay
        self._weight_decay = weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

        self._base_lr = learning_rate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        lr = self._lr
        t = tf.cast(global_step, dtype=tf.float32)

        if self._decay > 0.:
            lr *= (1. / (1. + self._decay * t))

        t += 1

        bias_correction1 = 1. - (self._beta1 ** t)
        bias_correction2 = 1. - (self._beta2 ** t)

        # Compute the maximum length of the approximated SMA
        p_inf = 2. / (1. - self._beta2) -  1.

        assignments = []
        for grad, param in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/radam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/radam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            v_t = (
                    tf.multiply(self._beta2, v) + tf.multiply(1. - self._beta2, tf.square(grad)))
            m_t = (
                    tf.multiply(self._beta1, m) + tf.multiply(1. - self._beta1, grad))
            m_t_hat = m_t / bias_correction1

            # Compute the length of the approximated SMA
            p_t = p_inf - 2. * t * (self._beta2 ** t) / bias_correction2

            if p_t > 4.:
                v_t_hat = tf.sqrt(v_t / bias_correction2)
                r_t = tf.sqrt(
                    (p_t - 4.) * (p_t - 2.) * p_inf / ((p_inf - 4.) * (p_inf - 2.) * p_t)
                )
                p_t = param - lr * r_t * m_t_hat / v_t_hat
            else:
                p_t = param - lr * m_t_hat

            if self._do_use_weight_decay(param_name):
                p_t += self._weight_decay * param

            update_list = [param.assign(p_t), m.assign(m_t), v.assign(v_t)]

            assignments.extend(update_list)

        # update the global step
        assignments.append(global_step.assign_add(1))

        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay:
            return False
        if self._exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

import numpy as np
import torch
import torch.nn.functional as F
import keras
from ..graph import *
from collections import defaultdict
from contextlib import contextmanager
from keras.backend.common import set_image_dim_ordering, image_dim_ordering
from keras.backend.common import floatx, epsilon, image_data_format


py_all = all
py_sum = sum



_LEARNING_PHASE = Tensor(name='keras_learning_phase')
_UID_PREFIXES = defaultdict(int)


def learning_phase():
    return _LEARNING_PHASE


def set_learning_phase(value):
    global _LEARNING_PHASE
    _LEARNING_PHASE = value


def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = defaultdict(int)


NAME_SCOPE_STACK = []


@contextmanager
def name_scope(name):
    global NAME_SCOPE_STACK
    NAME_SCOPE_STACK.append(name)
    yield
    NAME_SCOPE_STACK.pop()


def _prepare_name(name, default):
    prefix = '/'.join(NAME_SCOPE_STACK)
    if name is None:
        return prefix + '/' + default
    return prefix + '/' + name


def is_keras_tensor(x):
    return hasattr(x, '_keras_history')



def _is_num(x):
    try:
        float(x)
        return True
    except:
        return 'numpy' in str(type(x))


def _get_shape(x):
    if hasattr(x, 'value'):
        return x.value.size()
    if hasattr(x, 'shape'):
        return x.shape
    if hasattr(x, 'size'):
        return tuple(x.size())
    if _is_num(x):
        return ()
    return None


def make_keras_tensor(tensor, uses_learning_phase=False):
    tensor._keras_shape = int_shape(tensor)
    tensor._uses_learning_phase = uses_learning_phase


def variable(value, dtype=None, name=None, constraint=None):
    if isinstance(value, Tensor):
        value = value.value
    if isinstance(value, torch.autograd.Variable):
        value = value.data
    if 'torch' in str(type(value)):
        value = value.numpy()
    name = _prepare_name(name, 'variable')
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype != dtype:
        value = np.cast[dtype](value)
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=True)
    ktorch_variable = Variable(torch_variable, name=name)
    ktorch_variable.constraint = None
    make_keras_tensor(ktorch_variable)
    return ktorch_variable


def constant(value, dtype=None, shape=None, name=None):
    value = np.array(value)
    name = _prepare_name(name, 'constant')
    if dtype is None:
        dtype = keras.backend.floatx()
    if value.dtype != dtype:
        value = np.cast[dtype](value)
    if value.shape == ():
        if shape is None:
            shape = ()
        value = np.ones(shape) * value
    torch_tensor = torch.from_numpy(value)
    torch_variable = torch.autograd.Variable(torch_tensor, requires_grad=False)
    ktorch_variable = Variable(torch_variable, name=name)
    make_keras_tensor(ktorch_variable)
    return ktorch_variable


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    name = _prepare_name(name, 'placeholder')
    if sparse:
        raise Exception('Sparse tensors are not supported yet :( ')
    if dtype is None:
        dtype = keras.backend.floatx()
    ktorch_tensor = Tensor(name=name, shape=shape, ndim=ndim, dtype=dtype)
    make_keras_tensor(ktorch_tensor)
    ktorch_tensor._ktorch_placeholder = True
    return ktorch_tensor


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    """
    return hasattr(x, '_torch_placeholder') and x._torch_placeholder


def shape(x):
    if hasattr(x, 'value'):
        return Variable(tuple(x.value.size()))
    elif hasattr(x, 'shape'):
        return Variable(x.shape)
    else:
        raise Exception('Tensor shape not available.')


def int_shape(x):
    if hasattr(x, 'value'):
        return tuple(x.value.size())
    elif hasattr(x, 'shape'):
        return x.shape
    else:
        raise Exception('Tensor shape not available.')


def ndim(x):
    x_shape = _get_shape(x)
    if x_shape is None:
        return None
    else:
        return len(x_shape)


def dtype(x):
    if isinstance(x, Tensor):
        x = x.eval()
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    return type(x)


def eval(x):
    y = x.eval()
    if 'torch' in str(type(x)) and hasattr(y, 'data'):
        y = y.data
    if hasattr(y, 'numpy'):
        y = y.numpy()
    return y


def zeros(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.zeros(shape), dtype, name)


def ones(shape, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.ones(shape), dtype, name)


def eye(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    return variable(np.eye(size), dtype, name)


def ones_like(x, dtype=None, name=None):
    y = get_op(lambda x: x * 0. + 1.)(x)
    return y


def zeros_like(x, dtype=None, name=None):
    y = get_op(lambda x: x * 0.)(x)
    return y


def identity(x):
    y = get_op(lambda x: x + 0.)(x)
    return y


def count_params(x):
    return np.prod(x.eval().size())


def random_uniform_variable(shape, low, high, dtype=None, name=None):
    return variable(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype, name=name)


def cast(x, dtype):
    def _cast(x, dtype):
        return x.type(dtype)
    return get_op(_cast, arguments=[dtype])(x)

# UPDATES OPS


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def moving_average_update(variable, value, momentum):
    return (variable, variable * momentum + value * (1. - momentum))

def bias_add(x, bias, data_format=None):
    def _bias_add(X, data_format):
        x, bias = X
        from keras.backend import image_data_format, ndim, reshape
        if data_format is None:
            data_format = image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))
        if ndim(bias) != 1 and ndim(bias) != ndim(x) - 1:
            raise ValueError('Unexpected bias dimensions %d, '
                             'expect to be 1 or %d dimensions'
                             % (ndim(bias), ndim(x) - 1))
        bias_shape = tuple(bias.size())
        ndim_x = len(x.size())
        ndim_bias = len(bias_shape)
        if ndim_x == 5:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1, 1, 1))
                else:
                    bias = reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, 1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        elif ndim_x == 4:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1, 1))
                else:
                    bias = reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        elif ndim_x == 3:
            if data_format == 'channels_first':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, bias_shape[0], 1))
                else:
                    bias = reshape(bias, (1, bias_shape[1], bias_shape[0]))
            elif data_format == 'channels_last':
                if ndim_bias == 1:
                    bias = reshape(bias, (1, 1, bias_shape[0]))
                else:
                    bias = reshape(bias, (1,) + bias_shape)
        return x.add(bias.expand_as(x))

    def _compute_output_shape(X):
        return _get_shape(X[0])

    return get_op(_bias_add, output_shape=_compute_output_shape, arguments=[data_format])([x, bias])


def dot(x, y):
    def _dot(X):
        x, y = X
        x_ndim = ndim(x)
        y_ndim = ndim(y)
        if x_ndim == 2 and y_ndim == 2:
            return torch.mm(x, y)
        if x_ndim == 2 and y_ndim == 1:
            return torch.mv(x, y)
        if x_ndim == 1 and y_ndim == 2:
            return torch.mv(y, x)
        if x_ndim == 1 and y_ndim == 1:
            return torch.dot(x, y)
        else:
            raise Exception('Unsupported tensor ranks for dot operation : ' + str(x_ndim) + ' and ' + str(y_ndim) + '.')

    def _compute_output_shape(X):
        x, y = _get_shape(X[0]), _get_shape(X[1])
        x_ndim = len(x)
        y_ndim = len(y)
        if x_ndim == 2 and y_ndim == 2:
            return (x[0], y[1])
        if x_ndim == 2 and y_ndim == 1:
            return (x[0],)
        if x_ndim == 1 and y_ndim == 2:
            return (y[0],)
        if x_ndim == 1 and y_ndim == 1:
            return (0,)

    return get_op(_dot, output_shape=_compute_output_shape)([x, y])


def batch_dot(x, y, axes=None):
    if type(axes) is int:
        axes = (axes, axes)
    def _dot(X):
        x, y = X
        x_shape = x.size()
        y_shape = y.size()
        x_ndim = len(x_shape)
        y_ndim = len(y_shape)
        if x_ndim <= 3 and y_ndim <= 3:
            if x_ndim < 3:
                x_diff = 3 - x_ndim
                for i in range(diff):
                    x = torch.unsqueeze(x, x_ndim + i)
            else:
                x_diff = 0
            if y_ndim < 3:
                y_diff = 3 - y_ndim
                for i in range(diff):
                    y = torch.unsqueeze(y, y_ndim + i)
            else:
                y_diff = 0
            if axes[0] == 1:
                x = torch.transpose(x, 1, 2)
            elif axes[0] == 2:
                pass
            else:
                raise Exception('Invalid axis : ' + str(axes[0]))
            if axes[1] == 2:
                x = torch.transpose(x, 1, 2)
            # -------TODO--------------#


def transpose(x):
    def _transpose(x):
        dim_order = list(reversed(range(ndim(x))))
        return torch.Tensor.permute(x, *dim_order)

    def _compute_output_shape(X):
        return tuple(reversed(_get_shape(X)))

    return get_op(_transpose, output_shape=_compute_output_shape)(x)


# ELEMENT-WISE OPERATION

def max(x, axis=None, keepdims=False):
    def _max(x, axis, keepdims):
        y = torch.max(x, axis)[0]
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_max, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def min(x, axis=None, keepdims=False):
    def _min(x, axis, keepdims):
        y = torch.min(x, axis)[0]
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_min, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def sum(x, axis=None, keepdims=False):
    def _sum(x, axis, keepdims):
        y = torch.sum(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_sum, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def prod(x, axis=None, keepdims=False):
    def _prod(x, axis, keepdims):
        y = torch.prod(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_prod, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def std(x, axis=None, keepdims=False):
    def _std(x, axis, keepdims):
        y = torch.std(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_std, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def var(x, axis=None, keepdims=False):
    def _var(x, axis, keepdims):
        y = torch.var(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis, keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_var, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def cumsum(x, axis=0):
    def _cumsum(x, axis=axis):
        y = torch.cumsum(x, axis)
        return y

    def _compute_output_shape(x, axis=axis):
        return _get_shape(x)

    return get_op(_cumsum, output_shape=_compute_output_shape, arguments=[axis])(x)

#~~~~~~~~~~~~~~ UNIMPLEMENTED IN PYTORCH !! ~~~~~~~~~~~~~~#


def cumprod(x, axis=0):
    def _cumprod(x, axis=axis):
        y = torch.cumprod(x, axis)
        return y

    def _compute_output_shape(x, axis=axis):
        return _get_shape(x)

    return get_op(_cumprod, output_shape=_compute_output_shape, arguments=[axis])(x)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def mean(x, axis=None, keepdims=False):
    def _mean(x, axis=axis, keepdims=keepdims):
        y = torch.mean(x, axis)
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis, keepdims=keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_mean, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def any(x, axis=None, keepdims=False):
    def _any(x, axis=axis, keepdims=keepdims):
        y = torch.sum(x != 0, axis) != 0
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis, keepdims=keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_any, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def all(x, axis=None, keepdims=False):
    def _all(x, axis=axis, keepdims=keepdims):
        y = torch.sum(x == False, axis) == 0
        # Since keepdims argument of torch not functional
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis, keepdims=keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_all, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def argmax(x, axis=-1):
    def _argmax(x, axis=axis):
        y = torch.max(x, axis)[1]
        # Since keepdims argument of torch not functional
        return torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis):
        shape = list(_get_shape(x))
        del shape[axis]

        return tuple(shape)

    return get_op(_argmax, output_shape=_compute_output_shape, arguments=[axis])(x)


def argmin(x, axis=-1):
    def _argmin(x, axis=axis):
        y = torch.max(x, axis)[1]
        # Since keepdims argument of torch not functional
        return torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis):
        shape = list(_get_shape(x))
        del shape[axis]

        return tuple(shape)

    return get_op(_argmin, output_shape=_compute_output_shape, arguments=[axis])(x)


def square(x):
    y = get_op(lambda x: x * x)(x)
    return y


def abs(x):
    y = get_op(lambda x: torch.abs(x))(x)
    return y


def sqrt(x):
    y = get_op(lambda x: torch.sqrt(x))(x)
    return y


def exp(x):
    y = get_op(lambda x: torch.exp(x))(x)
    return y


def log(x):
    y = get_op(lambda x: torch.log(x))(x)
    return y


def logsumexp(x, axis=None, keepdims=False):
    def _logsumexp(x, axis=axis, keepdims=keepdims):
        y = torch.log(torch.sum(torch.exp(x), axis))
        return y if keepdims else torch.squeeze(y, axis)

    def _compute_output_shape(x, axis=axis, keepdims=keepdims):
        if axis is None:
            return ()

        shape = list(_get_shape(x))
        if keepdims:
            shape[axis] = 1
        else:
            del shape[axis]

        return tuple(shape)

    return get_op(_logsumexp, output_shape=_compute_output_shape, arguments=[axis, keepdims])(x)


def round(x):
    y = get_op(lambda x: torch.round(x))(x)
    return y


def sign(x):
    y = get_op(lambda x: torch.sign(x))(x)
    return y


def pow(x, exp):
    def _pow(x, exp=exp):
        return torch.pow(x, exp)

    return get_op(_pow, arguments=[exp])(x)


def clip(x, min_value, max_value):
    def _clip(x, min_value=min_value, max_value=max_value):
        if max_value is not None and max_value < min_value:
            max_value = min_value
        if max_value is None:
            max_value = np.inf

        return torch.clamp(x, min_value, max_value)

    return get_op(_clip, arguments=[min_value, max_value])(x)


def equal(x, y):
    def _equal(inputs):
        x, y = inputs
        return 1 - torch.clamp((torch.ceil(torch.abs(x - y))), 0, 1)

    return get_op(_equal)([x, y])


def not_equal(x, y):
    def _not_equal(inputs):
        x, y = inputs
        return torch.clamp((torch.ceil(torch.abs(x - y))), 0, 1)

    return get_op(_not_equal)([x, y])


def greater(x, y):
    def _greater(inputs):
        x, y = inputs
        return torch.ceil(torch.clamp((x - y), 0, 1))

    return get_op(_greater)([x, y])


def greater_equal(x, y):
    def _greater_equal(inputs):
        x, y = inputs
        return 1 + torch.clamp(torch.floor(x - y), -1, 0)

    return get_op(_greater_equal)([x, y])


def less(x, y):
    def _less(inputs):
        x, y = inputs
        return torch.ceil(torch.clamp(y - x, 0, 1))
    return get_op(_less)([x, y])


def less_equal(x, y):
    def _less_equal(inputs):
        x, y = inputs
        return torch.abs(torch.floor(torch.clamp((x - y), -1, 0)))

    return get_op(_less_equal)([x, y])


def maximum(x, y):
    def _maximum(inputs):
        return torch.max(inputs[0], inputs[1])

    return get_op(_maximum)([x, y])


def minimum(x, y):
    def _maximum(inputs):
        return torch.min(inputs[0], inputs[1])

    return get_op(_maximum)([x, y])


def sin(x):
    y = get_op(lambda x: torch.sin(x))(x)
    return y


def cos(x):
    y = get_op(lambda x: torch.cos(x))(x)
    return y


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    def _concatenate(tensors, axis=axis):
        return torch.cat(tensors, axis)

    def _compute_output_shape(tensors, axis=axis):
        new_axis = np.sum([_get_shape(tensor)[axis] for tensor in tensors])
        shape = list(_get_shape(tensors[0]))
        shape[axis] = new_axis
        return tuple(shape)

    return get_op(_concatenate, output_shape=_compute_output_shape, arguments=[axis])(tensors)


def reshape(x, shape):
    def _reshape(x, shape=shape):
        return x.view(shape)

    def _compute_output_shape(x, shape=shape):
        if -1 not in shape:
            return shape
        else:
            n_elems = np.prod(list(_get_shape(x)))
            new_shape = list(shape)
            new_shape.remove(-1)
            new_axis = n_elems // np.prod(new_shape)
            s = list(shape)
            s[s.index(-1)] = new_axis
            return tuple(s)

    return get_op(_reshape, output_shape=_compute_output_shape, arguments=shape)(x)


def permute_dimensions(x, pattern):
    def _permute_dimensions(x, pattern=pattern):
        return x.permute(*pattern)

    def _compute_output_shape(x, pattern=pattern):
        return tuple(np.asarray(_get_shape(x))[list(pattern)])

    return get_op(_permute_dimensions, output_shape=_compute_output_shape, arguments=[pattern])(x)


def arange(start, stop=None, step=1, dtype='int32'):
        #TODO : Other datatypes
        return torch.arange(start, stop, step).int()


def flatten(x):
    def _flatten(x):
        return x.view([-1])

    def _compute_output_shape(x):
        return (np.prod(list(_get_shape(x))),)

    return get_op(_flatten, output_shape=_compute_output_shape)(x)


def expand_dims(x, axis=-1):
    def _expand_dims(x, axis=axis):
        return torch.unsqueeze(x, axis)

    def _compute_output_shape(x, axis=axis):
        shape = list(_get_shape(x))
        shape.insert(axis, 1)
        return shape

    return get_op(_expand_dims, output_shape=_compute_output_shape, arguments=[axis])(x)


def squeeze(x, axis):
    def _squeeze(x, axis=axis):
        return torch.squeeze(x, axis)

    def _compute_output_shape(x, axis=axis):
        shape = list(_get_shape(x))
        del shape[axis]
        return shape

    return get_op(_squeeze, output_shape=_compute_output_shape, arguments=[axis])(x)


def stack(x, axis=0):
    def _stack(x, axis=axis):
        return torch.stack(x, axis)

    def _compute_output_shape(x, axis=axis):
        n = len(x)
        shape = list(_get_shape(x[0]))
        shape.insert(axis, n)
        return shape

    return get_op(_stack, output_shape=_compute_output_shape, arguments=[axis])(x)


def one_hot(indices, num_classes):
    # Not differentiable
    def _one_hot(indices, num_classes=num_classes):
        temp = indices.view(-1,1).long().data
        batch_size = temp.size()[0]
        y = torch.zeros(batch_size, num_classes)
        return y.scatter_(1, temp, 1)

    return get_op(_one_hot, arguments=[num_classes])(indices)


# VALUE MANIPULATION

def get_value(x):
    return x.eval().data.numpy()


def batch_get_value(ops):
    return [x.eval().data.numpy() for x in ops]


def set_value(x, value):
    value = np.asarray(value)
    x.value.data = torch.from_numpy(value)


def batch_set_value(tuples):
    for x, value in tuples:
        set_value(x, value)


def get_variable_shape(x):
    return tuple(x.value.size())


def print_tensor(x, message=''):
    def _print_tensor(x, message=message):
        print(message, x.value.data)

    return get_op(_print_tensor, arguments=[message])(x)


## NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    def _relu(x, alpha=0., max_value=max_value):
        if alpha != 0.:
            negative_part = F.relu(-x)
        x = F.relu(x)

        if max_value is not None:
            print ("Meh")
            x = torch.clamp(x, max=max_value)

        if alpha != 0:
            x -= alpha * negative_part
        return x

    return get_op(_relu, arguments=[alpha, max_value])(x)


def elu(x, alpha=1.):
    def _elu(x, alpha=alpha):
        return F.elu(x)

    return get_op(_elu, arguments=[alpha])(x)


def softmax(x):
    def _softmax(x):
        return F.softmax(x)

    return get_op(_softmax)(x)


def softplus(x):
    def _softplus(x):
        return F.softplus(x)

    return get_op(_softplus)(x)


def softsign(x):
    def _softsign(x):
        return F.softsign(x)

    return get_op(_softsign)(x)


def sigmoid(x):
    def _sigmoid(x):
        return F.sigmoid(x)

    return get_op(_sigmoid)(x)


def hard_sigmoid(x):
    def _hard_sigmoid(x):
        x = (0.2 * x) + 0.5
        return torch.clamp(x, 0., 1.)

    return get_op(_hard_sigmoid)(x)


def tanh(x):
    def _tanh(x):
        return F.tanh(x)

    return get_op(_tanh)(x)


def dropout(x, level, noise_shape=None, seed=None):
    # No support for noise shape and seed as of now
    def _dropout(x, level=level):
        return F.dropout(x, p=level, training=True)

    return get_op(_dropout)(x)


def l2_normalize(x, axis):
    def _l2_normalize(x, axis):
        return torch.nn.functional.normalize(x, p=2, dim=axis)

    return get_op(_l2_normalize, arguments=[axis])(x)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.normal(mean, stddev, shape))


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.uniform(minval, maxval, shape))


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    #TODO dtype
    #TODO seed
    return torch.from_numpy(np.random.binomial(1, p, shape))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    x = random_normal(shape, mean, stddev, dtype, seed)
    return torch.clamp(x, mean - 2 * stddev, mean + 2 * stddev)
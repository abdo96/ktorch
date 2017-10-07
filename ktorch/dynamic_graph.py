from .graph.op import get_op
from .backend import int_shape
import patchy
import inspect


# allows if statements, python loops,
# torch built-ins etc in custom layers


def is_torched(layer_class):
    layer_class_name = layer_class.__name__
    attr = '_' + layer_class_name + '_torched'
    return hasattr(layer_class, attr)


def torch_layer(layer_class):
    if is_torched(layer_class):
        return
    layer_class._call = layer_class.call
    layer_class._compute_mask = layer_class.compute_mask
    class Dummy(object):

        def call(self, inputs, mask=None, **kwargs):
            call_fn = self._call
            if type(inputs) is list:
                shapes = [int_shape(x) for x in inputs]
            else:
                shapes = int_shape(inputs)
            output_shape = self.compute_output_shape(shapes)
            if mask is not None:
                kwargs['mask'] = mask
            op = get_op(call_fn, output_shape=lambda *_: output_shape, arguments=kwargs)
            return op(inputs)

    layer_class.call = Dummy.call
    #patchy.replace(layer_class.compute_mask, None, inspect.getsource(Dummy.compute_mask))
    #layer_class.compute_mask = Dummy.compute_mask
    layer_class_name = layer_class.__name__
    attr = '_' + layer_class_name + '_torched'
    setattr(layer_class, attr, True)


def torch_all_layers(globals):
    from keras.layers import Layer
    classes = globals.values()
    for c in classes:
        if type(c) is type:
            if issubclass(c, Layer):
                torch_layer(c)




class Dummy(object):
    def __call__(self, inputs, **kwargs):
        """Wrapper around self.call(), for handling internal references.

        If a Keras tensor is passed:
            - We call self._add_inbound_node().
            - If necessary, we `build` the layer to match
                the _keras_shape of the input(s).
            - We update the _keras_shape of every input tensor with
                its new shape (obtained via self.compute_output_shape).
                This is done as part of _add_inbound_node().
            - We update the _keras_history of the output tensor(s)
                with the current layer.
                This is done as part of _add_inbound_node().

        # Arguments
            inputs: Can be a tensor or list/tuple of tensors.
            **kwargs: Additional keyword arguments to be passed to `call()`.

        # Returns
            Output of the layer's `call` method.

        # Raises
            ValueError: in case the layer is missing shape information
                for its `build` call.
        """
        if isinstance(inputs, list):
            inputs = inputs[:]
        with K.name_scope(self.name):
            # Handle laying building (weight creating, input spec locking).
            if not self.built:
                # Raise exceptions in case the input is not compatible
                # with the input_spec specified in the layer constructor.
                self.assert_input_compatibility(inputs)

                # Collect input shapes to build layer.
                input_shapes = []
                for x_elem in _to_list(inputs):
                    if hasattr(x_elem, '_keras_shape'):
                        input_shapes.append(x_elem._keras_shape)
                    elif hasattr(K, 'int_shape'):
                        input_shapes.append(K.int_shape(x_elem))
                    else:
                        raise ValueError('You tried to call layer "' + self.name +
                                         '". This layer has no information'
                                         ' about its expected input shape, '
                                         'and thus cannot be built. '
                                         'You can build it manually via: '
                                         '`layer.build(batch_input_shape)`')
                if len(input_shapes) == 1:
                    self.build(input_shapes[0])
                else:
                    self.build(input_shapes)
                self.built = True

                # Load weights that were specified at layer instantiation.
                if self._initial_weights is not None:
                    self.set_weights(self._initial_weights)

            # Raise exceptions in case the input is not compatible
            # with the input_spec set at build time.
            self.assert_input_compatibility(inputs)

            # Handle mask propagation.
            def __collect_previous_mask(input_tensors):
                    masks = []
                    for x in input_tensors:
                        if hasattr(x, '_keras_history'):
                            inbound_layer, node_index, tensor_index = x._keras_history
                            node = inbound_layer.inbound_nodes[node_index]
                            mask = node.output_masks[tensor_index]
                            masks.append(mask)
                        else:
                            masks.append(None)
                    if len(masks) == 1:
                        return masks[0]
                    return masks
            previous_mask = _collect_previous_mask(inputs)
            user_kwargs = copy.copy(kwargs)
            if not _is_all_none(previous_mask):
                # The previous layer generated a mask.
                if has_arg(self.call, 'mask'):
                    if 'mask' not in kwargs:
                        # If mask is explicitly passed to __call__,
                        # we should override the default mask.
                        kwargs['mask'] = previous_mask
            # Handle automatic shape inference (only useful for Theano).
            input_shape = _collect_input_shape(inputs)

            # Actually call the layer, collecting output(s), mask(s), and shape(s).
            from ktorch.dynamic_graph import torch_layer
            torch_layer(self.__class__)
            output = self.call(inputs, **kwargs)
            output_mask = self.compute_mask(inputs, previous_mask)
            #output_mask = get_op(self.compute_mask, arguments=[previous_mask])(inputs)
            # If the layer returns tensors from its inputs, unmodified,
            # we copy them to avoid loss of tensor metadata.
            output_ls = _to_list(output)
            inputs_ls = _to_list(inputs)
            output_ls_copy = []
            for x in output_ls:
                if x in inputs_ls:
                    x = K.identity(x)
                output_ls_copy.append(x)
            if len(output_ls_copy) == 1:
                output = output_ls_copy[0]
            else:
                output = output_ls_copy

            # Inferring the output shape is only relevant for Theano.
            if all([s is not None for s in _to_list(input_shape)]):
                output_shape = self.compute_output_shape(input_shape)
            else:
                if isinstance(input_shape, list):
                    output_shape = [None for _ in input_shape]
                else:
                    output_shape = None

            if not isinstance(output_mask, (list, tuple)) and len(output_ls) > 1:
                # Augment the mask to match the length of the output.
                output_mask = [output_mask] * len(output_ls)

            # Add an inbound node to the layer, so that it keeps track
            # of the call and of all new variables created during the call.
            # This also updates the layer history of the output tensor(s).
            # If the input tensor(s) had not previous Keras history,
            # this does nothing.
            self._add_inbound_node(input_tensors=inputs, output_tensors=output,
                                   input_masks=previous_mask, output_masks=output_mask,
                                   input_shapes=input_shape, output_shapes=output_shape,
                                   arguments=user_kwargs)

            # Apply activity regularizer if any:
            if hasattr(self, 'activity_regularizer') and self.activity_regularizer is not None:
                regularization_losses = [self.activity_regularizer(x) for x in _to_list(output)]
                self.add_loss(regularization_losses, _to_list(inputs))
        return output


def _patch_keras_engine():
    from keras.layers import Layer
    layer_call_fn = Layer.__call__
    patch_source = inspect.getsource(Dummy.__call__)
    patchy.replace(layer_call_fn, None, patch_source)


_KERAS_ENGINE_PATCHED = False

def initialize(globals):
    torch_all_layers(globals)
    if not _KERAS_ENGINE_PATCHED:
        _patch_keras_engine()

def call(self, x, mask, **kwargs):
	call_fn = self._call
	shape_fn = self.compute_output_shape
	if mask is not None:
		kwargs['mask'] = mask
	import ktorch
	op = ktorch.graph.op.get_op(call_fn, output_shape=shape_fn, arguments=kwargs)

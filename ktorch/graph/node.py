# Nodes allow greedy evaluation
from .tensor import Tensor

class Node(object):

	def __init__(self, inputs, outputs):
		if type(inputs) is not list:
			inputs = [inputs]
		if type(outputs) is not list:
			outputs = [outputs]
		self.inputs = [inp for inp in inputs if isinstance(inp, Tensor)]
		self.outputs = [op for op in outputs if isinstance(op, Tensor)]
		self.evaluated_inputs = [hasattr(inp, 'value') and inp.value for inp in self.inputs]
		if all(self.evaluated_inputs):
			for output in self.outputs:
				output.eval()
		for inp in self.inputs:
			inp.nodes.append(self)

	def ping(self, tensor):
		idx = self.inputs.index(tensor)
		if tensor.value is None:
			self.evaluated_inputs[idx] = False
			for output in self.outputs:
				if hasattr(output, 'value'):
					output.set_value(None)
			return
		self.evaluated_inputs[idx] = True
		if all(self.evaluated_inputs):
			for output in self.outputs:
				if hasattr(output, 'value'):
					del output.value
				output.eval()

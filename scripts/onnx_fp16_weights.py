"""Store large float32 weights as float16 and cast back to float32 inside the
graph. Halves the file size; compute stays float32 (only weight rounding)."""
import sys, onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
casts, new_inits = [], []
for init in g.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 256:
        arr = numpy_helper.to_array(init).astype(np.float16)
        h = numpy_helper.from_array(arr, init.name + "__fp16")
        new_inits.append(h)
        casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    else:
        new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.save(m, dst)
print("converted", len(casts), "tensors")

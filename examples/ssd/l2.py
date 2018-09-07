import caffe
from caffe import layers as L
import numpy as np

shape = [1, 32, 38, 38]
# prototxt
n = caffe.NetSpec()
n.data = L.Input(shape=dict(dim=shape))
n.conv4_3_norm = L.Normalize(n.data, scale_filler=dict(type="constant", value=20),
    across_spatial=False, channel_shared=False)
with open('l2.prototxt', 'w+') as f:
	print(n.to_proto(), file=f)

net = caffe.Net('l2.prototxt', caffe.TEST)
np.random.seed(1)
data = np.random.uniform(size=shape).astype(np.float32)
out = net.forward_all(**{"data": data})['conv4_3_norm']

data = data.reshape(shape[0] * shape[1], shape[2] * shape[3])
out = out.reshape(shape[0] * shape[1], shape[2] * shape[3])
np.savetxt('inputs.txt', data, fmt='%10.5f', delimiter=' ')
np.savetxt('outputs.txt', out, fmt='%10.5f', delimiter=' ')

print()
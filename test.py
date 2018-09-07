import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format as text_format
from caffe import layers as L
import caffe
from caffe.model_libs import *

model = caffe_pb2.NetParameter()
with open('models/MobileNet/mobilenet_deploy.prototxt', 'r') as f:
    text_format.Merge(f.read(), model)

resize_width = 510
resize_height = 300

train_transform_param = {
        'mirror': True,
        'mean_value': [130, 127, 125],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }

batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]

train_data = "examples/GTSDB/GTSDB_trainval_lmdb"
batch_size_per_device = 2
label_map_file = "data/GTSDB/labelmap_GTSDB.prototxt"

net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

# print(net.tops)
print(type(model.layer[0]))

for i in range(0, len(model.layer) - 1):
    net.layer.extend([model.layer[i]])

with open('net2.prototxt', 'w') as f:
	f.write(google.protobuf.text_format.MessageToString(net))

# def addLayerToNetParameter(layer, net_param, name, tops=[], bottoms=[]):
#     l = net_param.layer.add()
#     l.CopyFrom(layer.to_proto().layer[0])
#     l.name = name
#     for idx, top in enumerate(tops):
#         l.top[idx] = top
#     for bottom in bottoms:
#         l.bottom.add(bottom)

# # Create train net using net parameter
# net = caffe_pb2.NetParameter()
# net_data, net_label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
#         train=True, output_label=True, label_map_file=label_map_file,
#         transform_param=train_transform_param, batch_sampler=batch_sampler)

# addLayerToNetParameter(net_data, net, 'data', ['data', 'label'])

# body = caffe_pb2.NetParameter()
# with open(mobilenet_body, 'r') as f:
#     text_format.Merge(f.read(), body)

# for i in range(0, len(body.layer) - 1):
#     net.layer.extend([body.layer[i]])

# tmp_net = caffe.NetSpec()
# AddExtraLayers(tmp_net, use_batchnorm, lr_mult=lr_mult, from_layer=net.layer[-1].name)

# for idx, top in enumerate(tmp_net.tops):
#     addLayerToNetParameter(top, net, net.name, net.tops, net.bottoms)

# with open('net2.prototxt', 'w') as f:
#     f.write(pb.text_format.MessageToString(net))
# exit(0)

import torch

import networks


models = {}
models_pose = {}
parameters_to_train_pose = []



models["encoder"] = networks.LiteMono(model="lite-mono",drop_path_rate=0.2)
models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc,[0, 1, 2])

# models_pose["pose_encoder"] = networks.ResnetEncoder(18,"pretrained",num_input_images=2)
# parameters_to_train_pose += list(models_pose["pose_encoder"].parameters())
# models_pose["pose"] = networks.PoseDecoder(models_pose["pose_encoder"].num_ch_enc,
#                     num_input_features=1,
#                     num_frames_to_predict_for=2)
# print(models)
# print("--------")
# print(models_pose)
input0 = torch.ones(64, 3, 192, 640)
input1 = torch.ones(64, 3, 192, 640)

# pose_inputs = [models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
# axisangle, translation = models_pose["pose"](pose_inputs)
feature = models["encoder"](input0)
print(len(feature))
print(feature[0].shape)
print(feature[1].shape)
print(feature[2].shape)
print("----------")
# output = models["depth"](feature)
# print(len(output))
# print(type(output))
# # print(output)
# print(output[('disp', 2)].shape)
# print(output[('disp', 1)].shape)
# print(output[('disp', 0)].shape)
# print(output[0].shape)
# print(output[1].shape)
# print(output[2].shape)
import json
import numpy as np
import keras
from keras.models import load_model

# # splite weight to different files
# with open("weights_mnist.json") as json_file:
#     layers_w = json.load(json_file)
#     for layer_name in layers_w:
#         w = layers_w[layer_name]
#         if(len(w)>0):
#             with open("{}_weights.json".format(layer_name),'w') as save_file:
#                 json.dump(w[0], save_file)
#             save_file.close()
# json_file.close()

# # reshpa conv
# with open('conv2d_1_weights.json') as json_file:
#     w = json.load(json_file)
#     w = np.array(w)
#     w = np.swapaxes(w,0,3)
#     w = np.swapaxes(w, 1, 2)
#     swap_w = []
#     for k in range(len(w)):
#         aw = w[k]
#         aMatrix = []
#         print("aw",aw)
#         aw = aw.tolist()[0]
#         for i in range(len(aw)):
#             bw = aw[i]
#             print('bw',bw,i )
#             for j in range(len(bw)):
#                 cw = bw[j]
#                 print('cw',cw)
#                 aMatrix.append([i,j,cw])
#         swap_w.append(aMatrix)
#     with open('conv2d_1_w_swap.json','w') as save_file:
#         json.dump(swap_w, save_file)
#     save_file.close()
# json_file.close()

model = load_model('mnist_model.h5')
print(model.get_weights())

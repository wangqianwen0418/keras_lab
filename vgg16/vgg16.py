from __future__ import print_function
import keras
import json
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model
from keras import backend as K

from PIL import Image
from keras.preprocessing import image
img = Image.open('test.png')
# img = img.crop((0,0,224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(img.size, img.mode)
# print(x)

model_origi = VGG16(weights='imagenet', include_top=True)

outputs = {}
weights = {}
layer0 = model_origi.layers[0].input
for layer in model_origi.layers:
    model_inter = Model(inputs=[layer0], outputs=[layer.output])
    pred = model_inter.predict(x)
    outputs[layer.name]=pred[0].tolist()
    weights[layer.name]=[w.tolist() for w in layer.get_weights()]
    # print(pred[0].tolist())
with open("outputs_vgg.json","w") as json_file:
    json.dump(outputs, json_file)
json_file.close()
with open("weights_vgg.json","w") as json_file:
    json.dump(weights, json_file)
json_file.close()
print("finish")

import json
with open("weights_vgg.json")as json_file:
    data = json.load(json_file)
    for key in data:
        print(key)
json_file.close()
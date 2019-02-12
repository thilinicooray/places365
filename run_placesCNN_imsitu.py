import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import json

def get_model():
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model

def get_places365():
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    return classes

centre_crop = trn.Compose([
    trn.Resize((256,256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_place(img_dir, img_name, model, classes):
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    top1_pred = classes[idx[0]]

    return top1_pred

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument('--img_folder', type=str, default='./resized_256', help='Location of original images')
    args = parser.parse_args()

    model = get_model()
    model.eval()
    place_list = get_places365()

    img_pred_dict = {}

    image_path_folder = args.img_folder
    count = 0
    for file in os.listdir(image_path_folder):
        if file.endswith(".jpg"):
            print('processing image :', file)
            place_pred = predict_place(image_path_folder, file, model, place_list)
            img_pred_dict[file] = place_pred


    dir_name = os.path.basename(image_path_folder)
    with open(dir_name + '_imsitu_places.json', 'w') as fp:
        json.dump(img_pred_dict, fp)



if __name__ == "__main__":
    main()




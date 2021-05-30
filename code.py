import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import os
from shutil import copyfile
from collections import Counter
import csv
import sklearn.metrics
from transformers import ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MEAN = [0.5, 0.5, 0.5]
N_STD = [0.5, 0.5, 0.5]
IMAGE_SIZE = 384
print(device)

#**************KAGGLE DIRECTORIES************************
INPUT_DIR = '/kaggle/input/plant-pathology-2021-fgvc8/'
DATA_DIR = '/kaggle/input/'
WORKING_DIR = '/kaggle/working/'


def count_labels(file_path):
    with open(file_path, mode='r') as my_file:
        csv_reader = csv.DictReader(my_file)
        counter = Counter()
        for row in csv_reader:
            counter.update({row['labels']: 1})
        sorted_counter = sorted(counter.items())
        print(sorted_counter)
        print(len(sorted_counter))


class PreprocessedDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None, target_transform=None):
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        self.length = self.labels.size(0)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx].float()



def encode_topic(topic):
    topics = {
            'complex':              0,
            'frog_eye_leaf_spot':   1, 
            'healthy':              2, 
            'powdery_mildew':       3, 
            'rust':                 4, 
            'scab':                 5
            }

    word_list = topic.split()
    encoded = torch.zeros(6)
    for word in word_list:
        if word != 'healthy':
            encoded[topics[word]] = 1.
    return encoded

def decode_topic(topic):
    max_idx = torch.argmax(topic)
    topic = torch.sigmoid(topic).round().int()
    topics = [
            'complex',
            'frog_eye_leaf_spot',
            'healthy',
            'powdery_mildew',
            'rust',
            'scab'
            ]
    text = ''
    for i, val in enumerate(topic):
        if val == 1:
            text += topics[i] + ' '

    if text == '':
        text = topics[max_idx]
    
    return text

#initialize various pretrained models
def init_model(model_name):
    num_classes = 6
    model = None

    if model_name == 'vgg':
        model = models.vgg11_bn(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(pretrained=True)
        for params in model.parameters():
            params.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        for params in model.parameters(): 
            params.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        for params in model.parameters(): 
            params.requires_grad = False
        num_ftrs = model.classifier.in_features 
        model.classifier = nn.Linear(num_ftrs, num_classes) 
    elif model_name == "vit384":
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
        for params in model.parameters(): 
            params.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes) 
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model



def train(model, loader, criterion, optimizer):
    model.train()
    average_loss = 0.0
    average_f1 = 0.0
        
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        outputs = append_healthy(torch.round(torch.sigmoid(outputs)).int())
        labels = append_healthy(torch.round(labels).int())

        average_loss += loss.item() * labels.size(0) / len(loader.dataset)
        average_f1 += sklearn.metrics.f1_score(labels.to('cpu'), outputs.to('cpu'), average="samples") * labels.size(0) / len(loader.dataset)
        torch.cuda.empty_cache()

    return average_loss, average_f1

def evaluate(model, loader, criterion, optimizer):
    model.eval()
    average_loss = 0.0
    average_f1 = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

        outputs = torch.sigmoid(outputs)
        outputs = append_healthy(torch.round(outputs).int())
        labels = append_healthy(torch.round(labels).int())

        average_loss += loss.item() * labels.size(0) / len(loader.dataset)
        average_f1 += sklearn.metrics.f1_score(labels.to('cpu'), outputs.to('cpu'), average="samples") * labels.size(0) / len(loader.dataset)
        torch.cuda.empty_cache()

    return average_loss, average_f1



def collate_train_batch(batch):
    transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(), 
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
    images = []
    labels = []
    for image, label in batch:
        images.append(transform(image).unsqueeze(0))
        labels.append(label.unsqueeze(0))

    return torch.cat(images), torch.cat(labels)

def collate_valid_batch(batch):
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE)])
    images = []
    labels = []
    for image, label in batch:
        images.append(transform(image).unsqueeze(0))
        labels.append(label.unsqueeze(0))

    return torch.cat(images), torch.cat(labels)

def add_grad(model, layer_name):
    l = len(layer_name)
    for name, parameters in model.named_parameters():
        if name[0:l] == layer_name:
            parameters.requires_grad = True

def train_model_processed(epochs, offset, model_name, version):
    torch.cuda.empty_cache()
    model = init_model(model_name)

    model = model.to(device)

    best_valid_f1 = [0.]

    batch_size = 32
    lr = 0.003

    for epoch in range(0, epochs):
        torch.cuda.empty_cache()

        for name, parameters in model.named_parameters():
            parameters.requires_grad = False

        add_grad(model, 'classifier')

        if epoch >= 10:
            batch_size = 8
            lr = 0.001
            for name, parameters in model.named_parameters():
                parameters.requires_grad = True


        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()

        print('Epoch ' + str(epoch + offset + 1))

        train_loss = 0.
        train_f1 = 0.

        #dataset is split into 10 smaller datasets due to memory limitations
        #datasets 0 - 8 are used for training
        #dataset 9 is used for validation below
        for i in range(0, 9):
            train_data = PreprocessedDataset(DATA_DIR + '384modfive/384-mod/images_matrix_384px_' + str(i) + '.pt', 
                                            DATA_DIR + '384modfive/384-mod/labels_matrix_384px_' + str(i) + '.pt',
                                    transform=transforms.Compose([transforms.ConvertImageDtype(torch.float), transforms.Normalize(N_MEAN, N_STD)]), 
                                    target_transform=transforms.Compose([transforms.ConvertImageDtype(torch.float)]))

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, collate_fn=collate_train_batch)

            this_loss, this_f1 = train(model, train_loader, criterion, optimizer)
            train_loss += this_loss/9.
            train_f1 += this_f1/9.


        print('{} Loss: {:.4f} Acc. {:.4f}'.format('training', train_loss, train_f1)) 

        valid_data = PreprocessedDataset(DATA_DIR + '384modfive/384-mod/images_matrix_384px_9.pt', DATA_DIR + '384modfive/384-mod/labels_matrix_384px_9.pt',
                                transform=transforms.Compose([transforms.ConvertImageDtype(torch.float), transforms.Normalize(N_MEAN, N_STD)]), 
                                target_transform=transforms.Compose([transforms.ConvertImageDtype(torch.float)]))

        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, collate_fn=collate_valid_batch)
        valid_loss, valid_f1 = evaluate(model, valid_loader, criterion, optimizer)
        torch.cuda.empty_cache()


        print('{} Loss: {:.4f} Acc. {:.4f}'.format('validation', valid_loss, valid_f1))

        with open("valid_acc_" + model_name + '_' + version + ".txt", "a") as myFile:
            myFile.write("Epoch " + str(epoch + offset + 1) + "\n")
            myFile.write("Training: " + str(train_loss) + ", " + str(train_f1.item()) + '\n')
            myFile.write("Validation: " + str(valid_loss) + ", " + str(valid_f1.item()) + '\n')
            myFile.write("**********************************************\n")

        torch.save(model, model_name + '_' + version + '_epoch' + str(epoch + offset + 1) + '.pth')
        best_valid_f1.append(valid_f1)


train_model_processed(25, 0, 'vit384', 'v1')


class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        for param in model1.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False
        for param in model3.parameters():
            param.requires_grad = False

    def forward(self, x):
        preds1 = self.model1(x.clone())
        preds2 = self.model2(x.clone())
        preds3 = self.model3(x.clone())

        return preds1 + preds2 + preds3

#load 3 different models for ensemble inference
def predict():
    torch.cuda.empty_cache()
    model1 = torch.load(DATA_DIR + 'densenetensemble2/densenet169_v2_epoch45.pth', map_location=device)
    model2 = torch.load(DATA_DIR + 'densenetensemble2/densenet169_v2_epoch48.pth', map_location=device)
    model3 = torch.load(DATA_DIR + 'densenetensemble2/densenet169_v2_epoch50.pth', map_location=device)
    model = Ensemble(model1, model2, model3)
    model.eval()
    model = model.to(device)

    toFloat = transforms.ConvertImageDtype(torch.float)
    normalize = transforms.Normalize(N_MEAN, N_STD)
    scale = transforms.Resize(IMAGE_SIZE) 
    crop = transforms.CenterCrop(IMAGE_SIZE)

    test_data = TestDataset(INPUT_DIR + 'test_images',  #TODO change back
                            transform=transforms.Compose([toFloat, normalize, scale, crop]))

    batch_size = 128 #TODO 256 works for 1 linear classifier vit384, 8 for local GPU
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1) #TODO change back

    my_file = open('submission.csv', 'w')
    my_file.write('image,labels')
    
    for i, (imgs, filenames) in enumerate(test_loader):
        torch.cuda.empty_cache()
        imgs = imgs.to(device)
        with torch.set_grad_enabled(False): #need this otherwise pytorch allocates a ton of memory for gradients
            preds = model(imgs) #TODO remove logits for ensemble

        for filename, pred in zip(filenames, preds):
            label = decode_topic(pred) #pass in raw floats (decode topic will apply sigmoid, etc)
            my_file.write('\n' + filename + ',' + label)

    my_file.close()


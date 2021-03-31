import torch.nn as nn
import torch
import torchvision.datasets as dset
import numpy as np
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import face_recognition
import torch.nn.functional as F
import imageio
import glob

def save_feature(path):
    root = glob.glob(path + '/*')

    for folder in root:
        image_folder = glob.glob(folder + '/*.jpg')
        images = [imageio.imread(file) for file in image_folder]

        for i, img in enumerate(images):
            face_location = face_recognition.face_locations(img)  # 얼굴 검출

            # 사진에서 검출된 여러 얼굴 중 가장 큰 얼굴 추출
            if face_location.__len__() > 0:
                y, w, h, x = face_location[0]
                max = img[y:h, x:w]
            else:
                raise Exception('얼굴 검출 실패')

            for loc in face_location:
                y, w, h, x = loc
                tmp = img[y:h, x:w]
                if max.shape[0] * max.shape[1] < tmp.shape[0] * tmp.shape[1]:
                    max = tmp

            face_encode = face_recognition.face_encodings(max)

            if len(face_encode) == 0:
                raise Exception('얼굴 검출 실패')
            else:
                np.save(folder + '/%d.npy' % i, face_encode)

def npy_loader(path):
    tmp=np.load(path)
    return torch.from_numpy(tmp).float()

def load_data(path):
    train_data = dset.folder.DatasetFolder(path, npy_loader, extensions='.npy')
    return train_data

def ComputeAccr(model, data_loader):
  model.eval()
  correct = 0
  total = 0
  for imgs, label in data_loader:
    imgs=imgs.squeeze(1)
    imgs = Variable(imgs).cuda()
    label = Variable(label).cuda()

    output = model(imgs)
    _,output_index = torch.max(output,1)
    total += label.size(0)
    correct += (output_index==label).sum().float()
    model.train()
    return 100*correct/total

def train_model(path, learning_rate, epoch):
    train_data = load_data(path)
    global class_list

    class_list = train_data.classes
    nclass = len(train_data.classes)

    train_batch = data.DataLoader(train_data, batch_size=5, shuffle=True)

    if torch.cuda.is_available():
        model = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,nclass)
        ).cuda()
    else:
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nclass)
        )

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        for imgs, labels in train_batch:
            imgs = imgs.squeeze(1)

            if torch.cuda.is_available():
                imgs = Variable(imgs, requires_grad=True).cuda()
                labels = Variable(labels).cuda()
            else:
                imgs = Variable(imgs, requires_grad=True)
                labels = Variable(labels)

            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

    return model

def predict(model,path):
    img = imageio.imread(path)
    pred_prob = []
    class_index = []

    face_location = face_recognition.face_locations(img)
    for loc in face_location:
        y, w, h, x = loc
        face = img[y:h, x:w]
        face_encode = face_recognition.face_encodings(face)

        if len(face_encode) > 0:
            face_encode = face_encode[0]
        else:
            continue

        face_encode = torch.tensor(face_encode).float()
        face_encode = face_encode.unsqueeze(0)

        if torch.cuda.is_available():
            face_encode = Variable(face_encode).cuda()
        else:
            face_encode = Variable(face_encode)

        output = model(face_encode)
        _, output_index = torch.max(output, 1)

        class_index.append(output_index)
        pred_prob.append(F.softmax(output[0], dim=0)[output_index])

    pred_prob=np.asarray(pred_prob)
    max_prob = pred_prob.max()

    return class_list[class_index[pred_prob.argmax()]], max_prob
import torch
import torch.nn as nn
from torch.utils import data
from mds189 import Mds189
import numpy as np
from skimage import io, transform
#import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import time
start = time.time()

# Helper functions for loading images.
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# flag for whether you're training or not
is_train = True
is_key_frame = True # TODO: set this to false to train on the video frames, instead of the key frames
model_to_load = 'model.ckpt' # This is the model to load during testing, if you want to eval a previously-trained model.

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# Parameters for data loader
params = {'batch_size': 128,  # TODO: fill in the batch size. often, these are things like 32,64,128,or 256
          'shuffle': True,
          'num_workers': 2
          }
# TODO: Hyper-parameters
num_epochs = 40
learning_rate = 7e-4
# NOTE: depending on your optimizer, you may want to tune other hyperparameters as well

# Datasets
# TODO: put the path to your train, test, validation txt files
if is_key_frame:
    # label_file_train =  '../data/hw6_mds189/keyframe_data_train.txt'
    # label_file_val  =  '../data/hw6_mds189/keyframe_data_val.txt'
    label_file_train =  '~/cs189/dataloader_files/keyframe_data_train.txt'
    label_file_val  = '~/cs189/dataloader_files/keyframe_data_val.txt'
    # NOTE: the kaggle competition test data is only for the video frames, not the key frames
    # this is why we don't have an equivalent label_file_test with keyframes
else:
    # label_file_train = '../data/hw6_mds189/videoframe_data_train.txt'
    # label_file_val = '../data/hw6_mds189/videoframe_data_val.txt'
    # label_file_test = '../data/hw6_mds189/videoframe_data_test.txt'
    label_file_train = '~/cs189/dataloader_files/videoframe_data_train.txt'
    label_file_val = '~/cs189/dataloader_files/videoframe_data_val.txt'
    label_file_test = '~/cs189/dataloader_files/videoframe_data_test.txt'

# TODO: you should normalize based on the average image in the training set. This shows
# an example of doing normalization
mean = [134.010302198 / 255, 118.599587912 / 255, 102.038804945 / 255]
std = [23.5033438916 / 255, 23.8827343458 / 255, 24.5498666589 / 255]
# TODO: if you want to pad or resize your images, you can put the parameters for that below.

# Generators
# NOTE: if you don't want to pad or resize your images, you should delete the Pad and Resize
# transforms from all three _dataset definitions.
train_dataset = Mds189(label_file_train,loader=default_loader,transform=transforms.Compose([
                                               # transforms.Pad(requires_parameters),    # TODO: if you want to pad your images
                                               # transforms.Resize(requires_parameters), # TODO: if you want to resize your images
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
train_loader = data.DataLoader(train_dataset, **params)

val_dataset = Mds189(label_file_val,loader=default_loader,transform=transforms.Compose([
                                               # transforms.Pad(),
                                               # transforms.Resize(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
val_loader = data.DataLoader(val_dataset, **params)

if not is_key_frame:
    test_dataset = Mds189(label_file_test,loader=default_loader,transform=transforms.Compose([
                                                   # transforms.Pad(),
                                                   # transforms.Resize(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)
                                               ]))
    test_loader = data.DataLoader(test_dataset, **params)
else:
    test_loader = val_loader

# TODO: one way of defining your model architecture is to fill in a class like NeuralNet()
# NOTE: you should not overwrite the models you try whose performance you're keeping track of.
#       one thing you could do is have many different model forward passes in class NeuralNet()
#       and then depending on which model you want to train/evaluate, you call that model's
#       forward pass. this strategy will save you a lot of time in the long run. the last thing
#       you want to do is have to recode the layer structure for a model (whose performance
#       you're reporting) because you forgot to e.g., compute the confusion matrix on its results
#       or visualize the error modes of your (best) model
class NeuralNetTest(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # you can define some common layers, for example:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 3, 5) # you should review the definition of nn.Conv2d online
        self.pool = nn.MaxPool2d(2, 2)
        # note: input_dimensions and output_dimensions are not defined, they
        # are placeholders to show you what arguments to pass to nn.Linear
        # self.fc1 = nn.Linear(input_dimensions, output_dimensions)
        self.fc1 = nn.Linear(3 * 222 * 110, 8)

    def forward(self, x):
        # now you can use the layers you defined, to write the forward pass, i.e.,
        # network architecture for your model
        x = self.pool(F.relu(self.conv1(x))) # x -> convolution -> ReLU -> max pooling
        # Tensors need to be reshaped before going into an fc layer
        # the -1 will correspond to the batch size
        # x = x.view(-1, num_elements_in_one_x_sample)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x)) # x -> fc (affine) layer -> relu
        return x

class NeuralNet1(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6 * 444 * 220
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 16 * 218 * 106
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 109 * 53, 16 * 53)
        self.fc2 = nn.Linear(16 * 53, 16 * 20)
        self.fc3 = nn.Linear(16 * 20, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # x -> convolution -> ReLU -> max pooling
        # 6 * 222 * 110
        x = self.pool(F.relu(self.conv2(x))) # x -> convolution -> ReLU -> max pooling
        # 16 * 109 * 53
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, stride = 2)
        self.conv2 = nn.Conv2d(12, 96, 3)
        self.conv3 = nn.Conv2d(96, 48, 3, 1)
#        self.conv4 = nn.Conv2d(384, 256, 3, 1)
#        self.conv5 = nn.Conv2d(256, 256, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 26 * 12, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 8)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # x -> convolution -> ReLU -> max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
#        x = F.relu(self.conv4(x))
#        x = self.pool(F.relu(self.conv5(x)))
#        x = nn.AdaptiveAvgPool2d((6,6))(x)
#        print(x.shape)
        num_feat = np.prod(x.shape[1:])
        x = x.view(-1, num_feat)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, 4)
        self.conv2 = nn.Conv2d(48, 128, 5)
        self.conv3 = nn.Conv2d(128, 192, 3, 1)
        self.conv4 = nn.Conv2d(192, 192, 3, 1)
#        self.conv5 = nn.Conv2d(256, 256, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(192 * 4 * 1, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 8)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # x -> convolution -> ReLU -> max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
#        x = self.pool(F.relu(self.conv5(x)))
#        x = nn.AdaptiveAvgPool2d((6,6))(x)
#        print(x.shape)
        num_feat = np.prod(x.shape[1:])
        x = x.view(-1, num_feat)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = NeuralNet().to(device)

# if we're only testing, we don't want to train for any epochs, and we want to load a model
if not is_train:
    num_epochs = 0
    model.load_state_dict(torch.load('model.ckpt'))

# Loss and optimizer
criterion =  nn.CrossEntropyLoss() #TODO: define your loss here. hint: should just require calling a built-in pytorch layer.
# NOTE: you can use a different optimizer besides Adam, like RMSProp or SGD, if you'd like
import adabound
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = adabound.AdaBound(model.parameters(), lr=learning_rate, final_lr = 0.1, weight_decay = 1e-5)

def accuracy(p, y):
#    print(p.shape, y.shape)
    p = p.cpu().argmax(dim=1).numpy()
    y = y.cpu().numpy()
    return np.sum(p == y)/len(p)

# Train the model
# Loop over epochs
print('Beginning training..')
total_step = len(train_loader)
loss_list, val_loss_list = [], []
for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))
    for i, (local_batch,local_labels) in enumerate(train_loader):
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        # Forward pass
        outputs = model.forward(local_ims)
        loss = criterion(outputs, local_labels)
        # TODO: maintain a list of your losses as a function of number of steps
        #       because we ask you to plot this information
        # NOTE: if you use Google Colab's tensorboard-like feature to visualize
        #       the loss, you do not need to plot it here. just take a screenshot
        #       of the loss curve and include it in your write-up.
        if i % 4 == 0:
            loss_list.append(loss.data.cpu().numpy().max())
        acc = accuracy(outputs, local_labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 4 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, TrainAcc: {}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), acc))
    with torch.no_grad():
        valacc = []
        mean_val_loss = []
        for local_batch, local_labels in val_loader:
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model.forward(local_ims)
            loss = criterion(outputs, local_labels)
            mean_val_loss.append(loss.data.cpu().numpy().max())
            valacc.append(accuracy(outputs, local_labels))
        print("Val Accuracy:", np.mean(valacc))
#        print(mean_val_loss)
        val_loss_list.append(np.mean(mean_val_loss))

end = time.time()
print('Time: {}'.format(end - start))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print('Beginning Testing..')
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    groundtruth_list = []
    for (local_batch,local_labels) in test_loader:
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = model.forward(local_ims)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        predicted_list.extend(predicted)
        groundtruth_list.extend(local_labels)
        correct += (predicted == local_labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Look at some things about the model results..
# convert the predicted_list and groundtruth_list Tensors to lists
pl = [p.cpu().numpy().tolist() for p in predicted_list]
gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

np.save("preds_b", pl)
np.save("true_labels_b", gt)

# TODO: use pl and gt to produce your confusion matrices

# view the per-movement accuracy
label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
for id in range(len(label_map)):
    print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))

# TODO: you'll need to run the forward pass on the kaggle competition images, and save those results to a csv file.

if not is_key_frame:
    # your code goes here!
    def results_to_csv(y_test, name = "submission.csv"):
        y_test = y_test.astype(int)
        y_test = [label_map[i] for i in y_test]
        df = pd.DataFrame({'Category': y_test})
        def img_lab(label):
            return str(x).zfill(4) + ".jpg"
        df.index = df.index.map(label)
        df.to_csv(name, index_label='Id')

    with torch.no_grad():
        preds = []
        for (local_batch, local_labels) in test_loader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            pred = model.forward(local_ims)
            _, pred2 = torch.max(pred.data, 1)
            preds.extend(list(pred2.cpu().numpy()))
        results_to_csv(np.array(preds))

np.save("train_loss_b", loss_list)
np.save("val_loss_b", val_loss_list)
# Save the model checkpoint
torch.save(model.state_dict(), 'model_b.ckpt')

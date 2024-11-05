import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from skimage import metrics


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


def showOrigDec(orig, noise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        imshow(orig[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy
        ax = plt.subplot(2, n, i + 1 + n)
        imshow(noise[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5, 0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.figtext(0.5, 0.5, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="r")
    plt.subplots_adjust(hspace=0.3)

    plt.show()


def showOrigNoiseOut(orig, noise, denoise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 6))

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        imshow(orig[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i + 1 + n)
        imshow(noise[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display denoised image
        ax = plt.subplot(3, n, i + 1 + n + n)
        imshow(denoise[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.figtext(0.5, 0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5, 0.65, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5, 0.35, " DENOISED RECONSTRUCTED IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# define the NN architecture
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        # Layer 1: is a convolutional layer that receives 3 channels of input and outputs num_features feature maps.
        # It is followed by the ReLU activation function.
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
            # Layer 2~16: each with a convolutional kernel size of 3×3, output num_features feature maps.
            # Each convolutional layer is followed by a batch normalisation and a ReLU activation function.
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        # layer 17: is a convolutional layer that changes the number of output channels back to 3
        # and is used to generate the residuals of the output image.
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        return y - residual


# convert data to a normalized torch.FloatTensor
transform = transforms.ToTensor()
# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)


num_workers = 0
noise_factor=0.1
batch_size = 32

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# print(torch.cuda.is_available())

'''
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# convert images to numpy for display
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
images = images.numpy()
noisy_imgs = noisy_imgs.numpy()
showOrigDec(images, noisy_imgs)
'''

# initialize the NN
model = DnCNN()
model.cuda()
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(type(train_loader))
for data in train_loader:
    images, target = data
    print(target)
    print(images.shape)
    break

# number of epochs to train the model
n_epochs = 40

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data

        ## add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        ## forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = model(noisy_imgs.cuda())
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = criterion(outputs, images.cuda())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * images.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))

torch.save(model, "denoise_DnCNN.pt")


model = torch.load('denoise_DnCNN.pt')
model.eval()

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# add noise to the test images
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

# get sample outputs
output = model(noisy_imgs.cuda())

# output is resized into a batch of iages
output = output.view(batch_size, 3, 32, 32)
# use detach when it's an output that requires_grad
output = output.detach().cpu()

dataiter = iter(test_loader)
images, labels = next(dataiter)

# add noise to the test images
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
showOrigNoiseOut(images, noisy_imgs, output)


avg_psnr=0
avg_ssim=0
test_size=0
for data in test_loader:
    images= data[0]
    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    output = model(noisy_imgs.cuda())
    output = output.view(len(images), 3, 32, 32)
    output = output.detach().cpu()
    batch_avg_psnr = 0
    batch_avg_ssim = 0
    for i in range(len(images)):
        org = np.transpose(images[i], (1, 2, 0)).numpy()
        denoise = np.transpose(output[i], (1, 2, 0)).numpy()
        batch_avg_psnr += metrics.peak_signal_noise_ratio(org, denoise, data_range=1.0)
        batch_avg_ssim += metrics.structural_similarity(org, denoise, multichannel=True, win_size=3, data_range=1.0)
    avg_psnr += batch_avg_psnr
    avg_ssim += batch_avg_ssim
    test_size += len(images)

print("On Test data of {} examples:\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(test_size, avg_psnr/test_size, avg_ssim/test_size))
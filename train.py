import torch as th
import torchvision as tv
import pro_gan_pytorch.PRO_GAN as pg

from torchvision import transforms
import torchvision
import argparse

#adding flags for training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help='Folder the dataset is in', required=True, type=str)
parser.add_argument('--batch', '-b', help='Batch size, or how many images in a grid', required=False, type=int, default='9')
parser.add_argument('--epochs', '-e', help='how many epochs to train for', required=False, type=int, default='10')
args = parser.parse_args()

# add the folder path for training here
folder = args.dataset
TRAIN_DATA_PATH = folder

# select the device to be used for training
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def setup_data(download=False):
    """
    setup the CIFAR-10 dataset for training the CNN
    :param batch_size: batch_size for sgd
    :param num_workers: num_readers for data reading
    :param download: Boolean for whether to download the data
    :return: classes, trainloader, testloader => training and testing data loaders
    """
    # data setup:
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(128),
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        #transforms.ToPILImage(mode='RGB'),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    
    testset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    
    classes = trainset.classes

    return classes, trainset, testset


if __name__ == '__main__':

    # some parameters:
    depth = 6
    # hyper-parameters per depth (resolution) default=10,10,20,20,20,20,20 can be higher for more detail. 
    # Change Batch size to get different amounts per grid. refers to the total number of images produced. default is 32.
    # amount needs to match depth parameter
    batch = args.batch
    epoch = args.epochs
    num_epochs = [epoch, epoch, epoch, epoch, epoch, epoch]
    fade_ins = [50, 50, 50, 50, 50, 50]
    batch_sizes = [batch, batch, batch, batch, batch, batch]
    latent_size = 128

    # get the data. Ignore the test data and their classes
    _, dataset, _ = setup_data(download=True)

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = pg.ConditionalProGAN(num_classes=len(dataset.classes), depth=depth, 
                                   latent_size=latent_size, device=device)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes
    )
    # ====================================================================== 
import os
import argparse
import torchvision
import torch

from torchvision import transforms
from torch.backends import cudnn
from solver import Solver
from DataLoader import ImageNet_val

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)


    print("\n==> Preparing data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # todo: change this path
    # data_set = ImageNet_val(
    #     image_path='/home/testaccount/ImageNet/val/', 
    #     xml_path='/home/testaccount/ImageNet/bbox/val/', 
    #     json_path='/home/testaccount/paper_workspace/all_test_script/T-resnet18/imagenet_class_index.json', 
    #     transform=transform
    # )

    # data_loader = torch.utils.data.DataLoader(
    #     data_set,
    #     batch_size=4
    # )

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size)

    # initialize solver
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        for epoch in range(0, config.num_epochs):
            solver.train(epoch)
    if config.mode == 'test':
        solver.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    parser.add_argument('--img_dir', type=str, default='./data/val/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--model_dir', type=str, default='./201809272208/')

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--G_num_filters', type=int, default=64)
    parser.add_argument('--D_num_filters', type=int, default=64)
    parser.add_argument('--G_num_bottleneck', type=int, default=6)
    parser.add_argument('--D_num_bottleneck', type=int, default=4)
    parser.add_argument('--G_lr', type=float, default=0.0001)
    parser.add_argument('--D_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=int, default=1)
    parser.add_argument('--lambda_diff', type=int, default=10)


    parser.add_argument('--D_steps', type=int, default=10)
    # parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--save_step', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)

    config = parser.parse_args()
    print(config)
    main(config)

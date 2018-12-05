import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms

from torch.backends import cudnn
from tensorboardX import SummaryWriter 

from model import ResGenerator
from model import UNetGenerator
from model import Discriminator

from targets import *


class Solver(object):

    def __init__(self, imagenet_loader, config):

        self.imagenet_loader = imagenet_loader
        
        self.num_classes = config.num_classes
        self.G_num_filters = config.G_num_filters
        self.D_num_filters = config.D_num_filters
        self.G_num_bottleneck = config.G_num_bottleneck
        self.D_num_bottleneck = config.D_num_bottleneck
        self.G_lr = config.G_lr
        self.D_lr = config.D_lr
        
        self.lambda_cls = config.lambda_cls
        self.lambda_diff = config.lambda_diff
        
        self.batch_size = config.batch_size
        # self.num_iters = config.num_iters
        self.D_steps = config.D_steps
        self.save_step = config.save_step
        self.device = torch.device('cuda')

        self.model_dir = config.model_dir
        self.log_dir = config.log_dir
        
        # 1. prepare dataset
        self.data_loader = self.imagenet_loader
        self.writer = SummaryWriter()
        
        cudnn.benchmark = True

        # 2. model initialize
        print("\n==> Building model...")
        self.build_model()        
        self.T.eval()

        self.iter_index = 0
        self.adversarial_loss = torch.nn.BCELoss()
        self.classification_loss = torch.nn.CrossEntropyLoss()


    def build_model(self):
        self.G = ResGenerator(
            num_classes=self.num_classes,
            num_filters=self.G_num_filters,
            num_bottleneck=self.G_num_bottleneck
        )
        self.D = Discriminator(
            num_filters=self.D_num_filters,
            num_bottleneck=self.D_num_bottleneck
        )

        # use cifar10-trained-resnet18
        checkpoint = torch.load('./ResNeXt29_2x64d.t7')
        self.T = ResNeXt29_2x64d()
        self.T.load_state_dict(checkpoint['net'])

        # self.T = torchvision.models.resnet18(pretrained=True)
     
        self.G_optimizer = torch.optim.Adam(
            self.G.parameters(), 
            lr=self.G_lr, 
            betas=(0.5, 0.999)
        )
        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(),
            lr=self.D_lr,
            betas=(0.5, 0.999)
        )

        self.G.to(self.device)
        print('\n==> G complete..')
        self.D.to(self.device)
        print('\n==> D complete..') 
        self.T.to(self.device)
        print('\n==> T complete..') 



    def save_model(self, file_name):
        G_dir = os.path.join(self.model_dir, '{}_G.pth'.format(file_name))
        D_dir = os.path.join(self.model_dir, '{}_D.pth'.format(file_name))

        torch.save(self.G.state_dict(), G_dir)
        torch.save(self.D.state_dict(), D_dir)

        print('\n==> Saved Model In {}'.format(self.model_dir))

    def to_onehot(self, index):
        size = index.shape[0]
        index = index.view(size, 1)
        onehot = torch.FloatTensor(size, self.num_classes)
        onehot = onehot.zero_()
        onehot = onehot.scatter_(1, index, 1)
        return onehot

    def reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

#     def classification_loss(self, out, label):
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(out, label)
#         return loss

    def train(self, epoch):
        start_iter = 1
        print('\nEpoch: %d' % epoch)
        step_timer = time.time()

        for batch_idx, (ori_image, ori_label) in enumerate(self.data_loader):
            

            target_index = torch.LongTensor(ori_image.shape[0]).random_(self.num_classes)
            target_label = self.to_onehot(target_index)
            soft_real = torch.Tensor(ori_image.shape[0]).uniform_(0.9, 1.0)
            soft_fake = torch.Tensor(ori_image.shape[0]).uniform_(0.0, 0.1)
            real = torch.Tensor(ori_image.shape[0]).fill_(1.0)
            
            # .. process input
            ori_image = ori_image.to(self.device)
            # ori_label = ori_label.to(self.device)           # what for?
            target_index = target_index.to(self.device)     # target index to compute cls_loss
            target_label = target_label.to(self.device)     # target label to generate mask_vector
            soft_real = soft_real.to(self.device)
            soft_fake = soft_fake.to(self.device)
            real = real.to(self.device)


            # train D
            if (batch_idx + 1) % self.D_steps != 0:
                d_out = self.D(ori_image)
                # d_loss_real = torch.log(d_out).mean()
                d_loss_real = self.adversarial_loss(d_out, soft_real)
            
                gen_image = self.G(ori_image, target_label)
                d_out = self.D(gen_image.detach())
                # d_loss_fake = torch.log(0.9 - d_out).mean()           # replace 1 to 0.999 to avoid NaN output
                d_loss_fake = self.adversarial_loss(d_out, soft_fake)
            

                # d_loss = - d_loss_real - d_loss_fake
                d_loss = (d_loss_real + d_loss_fake) / 2
                # print('==> d_loss: ' + str(d_loss.item()))
            
                self.reset_grad()
                d_loss.backward()
                self.D_optimizer.step()

                # logging
                self.writer.add_scalar('D/d_loss', d_loss, self.iter_index) 
                self.writer.add_scalar('D/d_loss_real', d_loss_real, self.iter_index) 
                self.writer.add_scalar('D/d_loss_fake', d_loss_fake, self.iter_index)

                real_image = utils.make_grid(ori_image)
                fake_image = utils.make_grid(gen_image)
                self.writer.add_image('D/real_image', real_image, self.iter_index)
                self.writer.add_image('D/fake_image', fake_image, self.iter_index)
                
                del real_image, fake_image, gen_image
                del d_out, d_loss_fake, d_loss_real

            # train G
            else:
                gen_image = self.G(ori_image, target_label)
                d_out = self.D(gen_image)
                # g_loss_fake = torch.log(0.9 - d_out).mean()           # replace 1 to 0.999 to avoid NaN output
                g_loss_fake = self.adversarial_loss(d_out, real)
                # print('==> g_loss_fake: ' + str(g_loss_fake.item()))
                                      
                # d_out = self.D(ori_image)
                # g_loss_real = torch.log(d_out).mean()
                # print('==> g_loss_real: ' + str(g_loss_real.item()))
                      
                # check classification result
                gen_class = self.T(gen_image)
                g_loss_cls = self.classification_loss(gen_class, target_index)
                # print('==> g_loss_cls: ' + str(g_loss_cls.item()))

                g_loss_diff = torch.mean(torch.abs(gen_image - ori_image))
                # print('==> g_loss_diff: ' + str(g_loss_diff.item()))

                # g_loss = g_loss_real + g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_diff * g_loss_diff
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_diff * g_loss_diff
                # print('==> g_loss: ' + str(g_loss.item()))

                self.reset_grad()
                g_loss.backward()
                self.G_optimizer.step()

                # logging
                self.writer.add_scalar('G/g_loss', g_loss, self.iter_index)
                self.writer.add_scalar('G/g_loss_cls', g_loss_cls, self.iter_index)
                self.writer.add_scalar('G/g_loss_diff', g_loss_diff, self.iter_index)
                # writer.add_scalar('G/g_loss_real', g_loss_real, self.iter_index)
                self.writer.add_scalar('G/g_loss_fake', g_loss_fake, self.iter_index)

                real_image = utils.make_grid(ori_image)
                fake_image = utils.make_grid(gen_image)
                self.writer.add_image('G/real_image', real_image, self.iter_index)
                self.writer.add_image('G/fake_image', fake_image, self.iter_index)
                
                del gen_image, ori_image
                del d_out, g_loss_fake, g_loss_diff


            # outputs
            if (batch_idx + 1) % 10 == 0:
                step_time = time.time() - step_timer
                step_timer = time.time()
                print("==> Step: {}\tTime: {:.4f}\tD_Loss: {:.6f}\tG_loss: {:.6f}\tG_loss_cls: {:.6f}".format(self.iter_index, step_time, d_loss, g_loss, g_loss_cls))

            
            self.iter_index = self.iter_index + 1
            # todo: learing_rate decay

        
        # save checkpoint
        if (epoch%20)==0:
            self.save_model(epoch)      
            


    def test(parameter_list):
        pass


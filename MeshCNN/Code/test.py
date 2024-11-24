import os
import time
import torch
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import DataLoader
from util.writer import Writer
from os.path import join
from util.util import seg_accuracy, print_network
from models import networks
import torch.nn as nn
import torch.optim as optim

class ClassifierModel:
    """ Class for training Model weights """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device_0 = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features_0 = None
        self.labels_0 = None
        self.mesh_0 = None
        self.soft_label_0 = None
        self.loss = None

        self.nclasses = opt.nclasses

        self.net_0 = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                                [self.gpu_ids[0]], opt.arch, opt.init_type, opt.init_gain)

        self.net_0.train(self.is_train)
        self.criterion_0 = networks.define_loss(opt).to(self.device_0)

        if self.is_train:
            self.optimizer_0 = torch.optim.Adam(self.net_0.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler_0 = networks.get_scheduler(self.optimizer_0, opt)
            print_network(self.net_0)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()

        self.edge_features_0 = input_edge_features.to(self.device_0).requires_grad_(self.is_train)
        self.labels_0 = labels.to(self.device_0)
        self.mesh_0 = data['mesh']

        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label_0 = torch.from_numpy(data['soft_label'])

    def forward(self):
        return self.net_0(self.edge_features_0, self.mesh_0)

    def backward(self, out):
        self.loss = self.criterion_0(out, self.labels_0)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer_0.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer_0.step()

    def load_network(self, which_epoch):
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net_0 = self.net_0
        if isinstance(net_0, torch.nn.DataParallel):
            net_0 = net_0.module
        print('loading the model from %s' % load_path)
        state_dict_0 = torch.load(load_path, map_location=str(self.device_0))
        if hasattr(state_dict_0, '_metadata'):
            del state_dict_0._metadata
        
        # Remove 'module.' prefix from keys if present
        new_state_dict_0 = {key[7:] if key.startswith('module.') else key: value for key, value in state_dict_0.items()}
        
        # Load the modified state_dict into the model
        net_0.load_state_dict(new_state_dict_0)

    def save_network(self, which_epoch):
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net_0.cpu().state_dict(), save_path)
            self.net_0.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net_0.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        self.scheduler_0.step()
        lr_0 = self.optimizer_0.param_groups[0]['lr']
        print('learning rate GPU 0 = %.7f' % lr_0)

    def run_test(epoch=-1):
        print('Running Test')
        opt = TestOptions().parse()
        opt.serial_batches = True  # no shuffle
        dataset = DataLoader(opt)
        classifier_model = ClassifierModel(opt)
        writer = Writer(opt)
        # test
        writer.reset_counter()
        for i, data in enumerate(dataset):
            classifier_model.set_input(data)
            ncorrect, nexamples = classifier_model.test()
            writer.update_counter(ncorrect, nexamples)
        writer.print_acc(epoch, writer.acc)
        return writer.acc

    def test(self):
        with torch.no_grad():
            out = self.forward()
            pred_class = out.data.max(1)[1]
            label_class = self.labels_0
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])

if __name__ == '__main__':

    ClassifierModel.run_test()

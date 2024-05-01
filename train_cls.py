from dataset.dataset_kpt import build_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from tensorboardX import SummaryWriter
from models.ResNet import resnet50
from utils import misc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    args = config.args
    save_path = './experiments/'
    model_name = 'train'
    print(model_name)
    if model_name != 'test':
        if not os.path.exists('./experiments/param/' + model_name + '/'):
            os.makedirs('./experiments/conf_matrix/' + model_name)
            os.makedirs('./experiments/weights/' + model_name)
            os.makedirs('./experiments/savept/' + model_name)
            os.makedirs('./experiments/param/' + model_name)
    device = torch.device('cuda')
    # data info
    Train_Dataset = build_dataset(args.dataset_root_kpt, True)
    train_loader = DataLoader(dataset=Train_Dataset, batch_size=32,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)
    train_size = int(Train_Dataset.__len__())

    Test_Dataset = build_dataset(args.dataset_root_kpt, False)
    test_loader = DataLoader(dataset=Test_Dataset, batch_size=32,
                             num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_size = int(Test_Dataset.__len__())
    print('model_name: ', model_name)
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    num_class = 8
    a = 0.5
    model = resnet50(num_classes=num_class)
    model = model.cuda()
    loss_ce = nn.CrossEntropyLoss().cuda()
    loss_l1 = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    best_acc_train = 0.0
    best_acc_test = 0.0
    temp_test = 0

    for i in range(500):
        print('epoch: ', i)
        losses = 0
        correct_train = 0
        conf_matrix = [[0 for _ in range(num_class)] for _ in range(num_class)]
        model.train()
        for kpt, label in tqdm(train_loader):
            kpt = kpt.cuda()
            label = label.cuda()
            output = model(kpt)
            loss = loss_ce(output, label)
            prediction = output.data.max(1)[1]
            correct_train += prediction.eq(label.data).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        lr_scheduler.step()
        train_acc = 100 * float(correct_train) / train_size
        best_acc_train, _ = misc.log_save_kpt(model_name, train_writer, train_logger, i, losses,
                                                         train_size, best_acc_train, train_acc, conf_matrix, mode=True)
        loss_dict = {}
        losses = 0
        correct_test = 0
        model.eval()
        for kpt, label in tqdm(test_loader):
            with torch.no_grad():
                kpt = kpt.cuda()
                label = label.cuda()
                output = model(kpt)
                loss = loss_ce(output, label)
                prediction = output.data.max(1)[1]
                conf_matrix = misc.get_conf_matrix(prediction, label, conf_matrix)
                correct_test += prediction.eq(label.data).sum()
                losses += loss.item()
        test_acc = 100 * float(correct_test) / test_size
        best_acc_test, is_save = misc.log_save_kpt(model_name, test_writer, test_logger, i, losses,
                                                   test_size, best_acc_test, test_acc, conf_matrix, mode=False)
        if is_save:
            torch.save(model.state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')

    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)

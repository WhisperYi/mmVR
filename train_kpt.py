# from dataset.dataset_lits_val import Val_Dataset
from dataset.datasets import build_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from tensorboardX import SummaryWriter
from models.mmVR_Transformer import build_model
from utils import misc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

if __name__ == '__main__':
    args = config.args
    save_path = './experiments/'
    model_name = 'train'
    if model_name != 'test':
        if not os.path.exists('./experiments/param/' + model_name + '/'):
            os.makedirs('./experiments/conf_matrix/' + model_name)
            os.makedirs('./experiments/weights/' + model_name)
            os.makedirs('./experiments/savept/' + model_name)
            os.makedirs('./experiments/param/' + model_name)
    device = torch.device('cuda')
    # data info
    Train_Dataset = build_dataset(args.dataset_root, args.mode)
    train_loader = DataLoader(dataset=Train_Dataset, batch_size=args.batch_size, collate_fn=misc.collate_fn,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)
    train_size = int(Train_Dataset.__len__())

    Test_Dataset = build_dataset(args.dataset_root, False)
    test_loader = DataLoader(dataset=Test_Dataset, batch_size=args.batch_size,collate_fn=misc.collate_fn,
                             num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_size = int(Test_Dataset.__len__())
    print('model_name: ', model_name)
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    model, criterion = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    best_mpjpe_train = 1000.0
    best_acc_train = 0.0
    best_mpjpe_test = 1000.0
    best_acc_test = 0.0
    temp_test = 0
    model = model.cuda()

    for i in range(args.epoch):
        misc.criterion_init(criterion)
        loss_dict = {}
        losses = 0
        model.train()
        criterion.train()
        for samples, imu, target in tqdm(train_loader):
            samples = samples.cuda()
            imu = imu.cuda()
            target = [{k: v.cuda() for k, v in t.items()} for t in target]
            out = model(samples, imu)
            loss_dict = criterion(out, target)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        best_mpjpe_train, best_acc_train, _, is_save = misc.log_save(model_name, train_writer, train_logger, i, losses, loss_dict, criterion, train_size, best_mpjpe_train, best_acc_train, mode=True)

        misc.criterion_init(criterion)
        loss_dict = {}
        losses = 0
        model.eval()
        criterion.eval()
        for samples, imu, target in tqdm(test_loader):
            with torch.no_grad():
                target = [{k: v.to(device) for k, v in t.items()} for t in target]
                samples = samples.to(device)
                imu = imu.cuda()
                out = model(samples, imu)
                loss_dict = criterion(out, target)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                # print(losses)
        best_mpjpe_test, best_acc_test, temp_test, is_save = misc.log_save(model_name, test_writer, test_logger, i,
                                                                           losses, loss_dict, criterion,
                                                                           test_size, best_mpjpe_test, best_acc_test,
                                                                           mode=False, temp_test=temp_test)
        if is_save:
            torch.save(model.state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')
    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)

#!/usr/bin/env python3
import argparse
import torch
from torch.optim.lr_scheduler import StepLR
from config import config
from net import net
from loss import trainLoss,testLoss
from dataloader import trainLoader,testLoader
from solver import solver
from csvlog import csvlog



def train(args, model, device, train_loader, optimizer, Loss, epoch):
    model.train()
    count=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        count+=len(data)
        if args.log_csv != "":
            args.csvlog((epoch-1)*len(train_loader.dataset)+count,loss.item(),None)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, Loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += Loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Training Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--traindata', type=str, default='train',
                        help='Path to train dataset (default: train)')
    parser.add_argument('--testdata', type=str, default='test',
                        help='Path to test dataset (default: test)')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.85, metavar='M',
                        help='Learning rate step gamma (default: 0.85)')
    parser.add_argument('--no-cuda', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='I',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load-model', default="",
                        help='For Loading pretrained weights - skip for none')
    parser.add_argument('--log-csv', default="",
                        help='Log train and test values into CSV file')
    parser.add_argument('--save-model-directory', default="snapshots/",
            help='Folder for Saving the current Model - empty string for none, (default: snapshots/)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg=config(args)

    train_loader = trainLoader(cfg,args)
    test_loader = testLoader(cfg,args)

    model = net(cfg,args,classes=train_loader.dataset.classes).to(device)
    train_loss = trainLoss(cfg,args)
    test_loss = testLoss(cfg,args)
    if args.load_model!="":
        model.load_state_dict(torch.load(args.load_model),strict=False)

    optimizer = solver(model.parameters(),cfg,args)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    if args.log_csv != "":
        args.csvlog=csvlog(args.log_csv)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, train_loss, epoch)
        tloss=test(model, device, test_loader, test_loss)
        if args.log_csv != "":
            args.csvlog((epoch)*len(train_loader.dataset),None,tloss)

        scheduler.step()

        if args.save_model_directory!="":
            torch.save(model.state_dict(), args.save_model_directory+("/epoch%03d.pt"%epoch))

    print("done");

if __name__ == '__main__':
    main()


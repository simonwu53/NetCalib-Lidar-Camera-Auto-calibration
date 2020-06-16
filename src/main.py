import argparse
import torch.optim as optim
import os
from train import load_checkpoint, run
from inference import run as run_test
from net.CalibrationNet import get_model
from net.utils import count_parameters
from DataLoader import get_dataloader
import logging
from config import ERR_GROUP, LR, L2, OPTIMIZER


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('Main')


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='Path to the saved model in saved folder.')
    parser.add_argument('--train', action='store_true', help='Start training process.')
    parser.add_argument('--test', action='store_true', help='Start testing process.')
    parser.add_argument('--err', type=int, default=4, help='Error group to select.')
    parser.add_argument('--iter', action='store_true', help='Using iterative calibration during testing.')
    parser.add_argument('--model', type=int, default=1, help='Select model variant to test.')
    parser.add_argument('--win-scale', type=float, default=2.5, help='Window size scaling while testing.')
    return parser.parse_args()


if __name__ == '__main__':
    # setup argparser
    args = arg_parser()

    # get model
    model = get_model(args.model)
    LOG.warning("Model trainable parameters: %d" % count_parameters(model))

    if args.train:
        # create optimizer
        if OPTIMIZER == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2, amsgrad=False)
        elif OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=LR, weight_decay=L2)
        else:
            raise NotImplementedError("Optimizer has not been implemented.")

        # load saved model
        if args.ckpt:
            epoch, global_step = load_checkpoint(model, args.ckpt, optimizer)
            save_root = os.path.dirname(os.path.dirname(args.ckpt))
            config = {'epoch': epoch, 'global_step': global_step,
                      'log_dir': os.path.join(save_root, 'log/'),
                      'save_dir': os.path.join(save_root, 'ckpt/'),
                      'save_root': save_root}
        else:
            config = {}
        model.cuda()

        # create datasets & loaders
        train_data, train_loader = get_dataloader(mode='train', err_dist='uniform',
                                                  err_config=ERR_GROUP[args.err])
        val_data, val_loader = get_dataloader(mode='val', err_dist='uniform',
                                              err_config=ERR_GROUP[args.err])

        # run training
        run(model, optimizer, (train_data, val_data), (train_loader, val_loader), config)

    elif args.test:
        if not args.ckpt:
            raise ValueError("No trained model given. Exit.")
        # load check point
        epoch, global_step = load_checkpoint(model, args.ckpt)

        # get saving root
        save_root = os.path.dirname(os.path.dirname(args.ckpt))

        # build config
        config = {'epoch': epoch, 'global_step': global_step,
                  'iter': args.iter if args.iter else False,
                  'log_dir': os.path.join(save_root, 'log/'),
                  'save_dir': os.path.join(save_root, 'ckpt/'),
                  'save_root': save_root,
                  'win_scale': args.win_scale}

        # change model to evaluation mode
        model.eval()
        model.cuda()

        # get dataset & loader
        val_data, val_loader = get_dataloader(mode='test', err_dist='uniform',
                                              err_config=ERR_GROUP[args.err])

        # run test
        run_test(model, val_data, val_loader, config)

    else:
        raise ValueError('--train or --test must be specified.')

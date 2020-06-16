from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import torch
from DataLoader import DataStat
from net.LidarProjection import proj_lidar
from config import *


def exp():
    dataset = DataStat()
    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    for i, data in enumerate(tqdm(loader)):
        depth, scan, T, P0, shape = data[0][0].numpy(), data[1][0].cuda(), data[2][0].cuda(), \
                                    data[3][0].cuda(), data[4][0].cuda()
        proj_pc = proj_lidar(scan, T, P0, resolution=shape, crop=CROP)

        cv2.imshow('Depth', depth)
        cv2.imshow('PC', proj_pc.cpu().numpy())
        keystroke = cv2.waitKey(50)
        if keystroke == ord('q') or keystroke == 27:  # q or ESC to quit
            cv2.destroyAllWindows()
            break
    return


def train_dataset_statistics(dataset):
    # create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                         pin_memory=True, drop_last=False)

    # statistics
    n_samples = len(dataset)
    n_pixels_per_frame = np.prod(CROP)
    x_sum_c = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x
    x2_sum_c = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # cam depth sum of x^2
    x_sum_l = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # lidar depth sum of x
    x2_sum_l = torch.tensor(0, dtype=torch.float32, device='cuda', requires_grad=False)  # lidar depth sum of x^2
    n_pixels_total = torch.tensor(n_samples*n_pixels_per_frame, dtype=torch.float32,
                                  device='cuda', requires_grad=False)  # total pixels

    # iterating over dataset
    for i, data in enumerate(tqdm(loader)):
        # unpack data
        cam_depth, lidar_pc, T, P0, shape = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4][0].cuda()

        # track the sum of x and the sum of x^2
        with torch.no_grad():
            lidar_depth = proj_lidar(lidar_pc[0], T[0], P0[0], shape, crop=CROP)
            x_sum_l += torch.sum(lidar_depth)
            x2_sum_l += torch.sum(lidar_depth**2)
            x_sum_c += torch.sum(cam_depth[0])
            x2_sum_c += torch.sum(cam_depth[0]**2)

    # calculate mean and std.
    # formula: stddev = sqrt((SUM[x^2] - SUM[x]^2 / n) / (n-1))
    mean_l = x_sum_l / n_pixels_total
    mean_c = x_sum_c / n_pixels_total
    std_l = torch.sqrt((x2_sum_l-x_sum_l**2/n_pixels_total)/(n_pixels_total-1))
    std_c = torch.sqrt((x2_sum_c-x_sum_c**2/n_pixels_total)/(n_pixels_total-1))
    print('Lidar Depth Map Statistics')
    print('Mean: %.4f' % mean_l.cpu().item())
    print('STD: %.4f' % std_l.cpu().item())
    print()
    print('Camera Depth Map Statistics')
    print('Mean: %.4f' % mean_c.cpu().item())
    print('STD: %.4f' % std_c.cpu().item())

    return mean_l, mean_c, std_l, std_c


if __name__ == '__main__':
    exp()

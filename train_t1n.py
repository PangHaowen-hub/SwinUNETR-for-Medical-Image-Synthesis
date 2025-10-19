from torch.utils.data import DataLoader
from tqdm import tqdm
import monai
import os
import torch
import torchio as tio
import logging
import time
from monai.inferers import SlidingWindowInferer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    train_data_root = r'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    val_data_root = r'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'

    train_txt = 'train.txt'
    val_txt = 'val.txt'

    name_list_train = []
    name_list_val = []

    with open(train_txt, "r") as f:
        for line in f.readlines():
            name_list_train.append(line.strip('\n'))

    with open(val_txt, "r") as f:
        for line in f.readlines():
            name_list_val.append(line.strip('\n'))

    input_0 = 't1c.nii.gz'
    input_1 = 't1n.nii.gz'
    input_2 = 't2f.nii.gz'
    input_3 = 't2w.nii.gz'

    train_0_paths = [os.path.join(train_data_root, i, i + '-' + input_0) for i in name_list_train]
    train_1_paths = [os.path.join(train_data_root, i, i + '-' + input_1) for i in name_list_train]
    train_2_paths = [os.path.join(train_data_root, i, i + '-' + input_2) for i in name_list_train]
    train_3_paths = [os.path.join(train_data_root, i, i + '-' + input_3) for i in name_list_train]

    val_0_paths = [os.path.join(val_data_root, i, i + '-' + input_0) for i in name_list_val]
    val_1_paths = [os.path.join(val_data_root, i, i + '-' + input_1) for i in name_list_val]
    val_2_paths = [os.path.join(val_data_root, i, i + '-' + input_2) for i in name_list_val]
    val_3_paths = [os.path.join(val_data_root, i, i + '-' + input_3) for i in name_list_val]

    subjects_train = []
    for (train_0_path, train_1_path, train_2_path, train_3_path) in zip(train_0_paths, train_1_paths, train_2_paths,
                                                                        train_3_paths):
        subject = tio.Subject(t1c=tio.ScalarImage(train_0_path), t1n=tio.ScalarImage(train_1_path),
                              t2f=tio.ScalarImage(train_2_path), t2w=tio.ScalarImage(train_3_path))
        subjects_train.append(subject)

    subjects_val = []
    for (val_0_path, val_1_path, val_2_path, val_3_path) in zip(val_0_paths, val_1_paths, val_2_paths, val_3_paths):
        subject = tio.Subject(t1c=tio.ScalarImage(val_0_path), t1n=tio.ScalarImage(val_1_path),
                              t2f=tio.ScalarImage(val_2_path), t2w=tio.ScalarImage(val_3_path))
        subjects_val.append(subject)
    transform = tio.Compose([tio.transforms.ZNormalization()])
    train_subjects = tio.SubjectsDataset(subjects_train, transform)
    val_subjects = tio.SubjectsDataset(subjects_val, transform)

    logger.info("Training set: %d subjects, Validation set: %d subjects" % (len(train_subjects), len(val_subjects)))

    patch_size = 128

    patches_training_set = tio.Queue(subjects_dataset=train_subjects, max_length=16, samples_per_volume=8,
                                     sampler=tio.data.UniformSampler(patch_size), num_workers=2, shuffle_subjects=True,
                                     shuffle_patches=True)
    return DataLoader(patches_training_set, batch_size=1), DataLoader(val_subjects, batch_size=1, pin_memory=True)


def train_and_test(net, trainloader, testloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5, mode='gaussian')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for images in tqdm(trainloader):
            t1c, t1n, t2f, t2w = images['t1c']['data'].to(DEVICE), images['t1n']['data'].to(DEVICE), images['t2f']['data'].to(DEVICE), images['t2w']['data'].to(DEVICE)
            image_input = torch.cat([t1c, t2f, t2w], dim=1)
            optimizer.zero_grad()
            outputs = net(image_input)
            loss = criterion(outputs, t1n)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        logger.info(f"Epoch {epoch + 1}: Train loss {epoch_loss / len(trainloader)}")

        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), os.path.join(now_time_path, "model_%d.pth" % (epoch + 1)))
            loss_test = 0.0
            criterion_val = torch.nn.L1Loss()
            net.eval()
            with torch.no_grad():
                for images in tqdm(testloader):
                    t1c, t1n, t2f, t2w = images['t1c']['data'].to(DEVICE), images['t1n']['data'].to(DEVICE), images['t2f']['data'].to(DEVICE), images['t2w']['data'].to(DEVICE)
                    image_input = torch.cat([t1c, t2f, t2w], dim=1)
                    pred = inferer(inputs=image_input, network=net)
                    loss_test += criterion_val(pred, t1n).item()
            logger.info(f"Epoch {epoch + 1}: Test loss {loss_test / len(testloader)}")


if __name__ == '__main__':
    load_model = None
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    now_time_path = os.path.join('experiments', now_time)
    os.makedirs(now_time_path, exist_ok=True)
    logger = logging.getLogger()
    logfile = '{}.log'.format(now_time)
    logfile = os.path.join(now_time_path, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

    net = monai.networks.nets.SwinUNETR(img_size=(128, 128, 128), in_channels=3, out_channels=1, depths=(2, 4, 2, 2))
    net.to(DEVICE)

    if load_model:
        print('Load model:' + load_model)
        net.load_state_dict(torch.load(load_model))

    trainloader, testloader = load_data()
    train_and_test(net, trainloader, testloader, epochs=100)

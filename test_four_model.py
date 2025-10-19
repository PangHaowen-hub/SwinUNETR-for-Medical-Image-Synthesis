import tqdm
from monai.inferers import SlidingWindowInferer
import os
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torchio as tio
import torch
import monai
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_one_case(val_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    file_list = os.listdir(val_folder)
    file_list.sort()
    full_modality_list = ['t1c', 't1n', 't2f', 't2w']
    modality_list = [file_list[0][-10:-7], file_list[1][-10:-7], file_list[2][-10:-7]]
    miss_one = list(set(full_modality_list) - set(modality_list))[0]
    print('the missing modality is: ' + miss_one)

    in_path_1 = os.path.join(val_folder, file_list[0])
    in_path_2 = os.path.join(val_folder, file_list[1])
    in_path_3 = os.path.join(val_folder, file_list[2])
    out_path = os.path.join(output_folder, file_list[2][:-10] + miss_one + file_list[2][-7:])

    modality_path_dict = {'t1c': '2024-07-17-23-40-48', 't1n': '2024-07-17-23-42-04', 't2f': '2024-07-17-23-42-46',
                          't2w': '2024-07-17-23-43-38'}
    experiment_path = os.path.join(r'./experiments', modality_path_dict[miss_one])
    model_name = 'model_100'
    model = monai.networks.nets.SwinUNETR(img_size=(128, 128, 128), in_channels=3, out_channels=1, depths=(2, 4, 2, 2)).cuda()
    model.load_state_dict(torch.load(os.path.join(experiment_path, model_name + '.pth')))

    subject = tio.Subject(m0=tio.ScalarImage(in_path_1), m1=tio.ScalarImage(in_path_2), m2=tio.ScalarImage(in_path_3))
    transform = tio.Compose([tio.transforms.ZNormalization()])
    val_subjects = tio.SubjectsDataset([subject], transform)
    test_dataloader = DataLoader(val_subjects, batch_size=1, pin_memory=True)

    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5, mode='gaussian')
    with torch.no_grad():
        model.eval()
        for images in test_dataloader:
            m0, m1, m2 = images['m0']['data'].to(DEVICE), images['m1']['data'].to(DEVICE), images['m2']['data'].to(DEVICE)
            image_input = torch.cat([m0, m1, m2], dim=1)
            pred = inferer(inputs=image_input, network=model)
            pred_np = pred.detach().cpu().squeeze().permute(2, 1, 0).numpy()

            m0_np = sitk.GetArrayFromImage(sitk.ReadImage(in_path_1))
            m1_np = sitk.GetArrayFromImage(sitk.ReadImage(in_path_2))
            m2_np = sitk.GetArrayFromImage(sitk.ReadImage(in_path_3))

            pred_np = pred_np - np.min(pred_np)
            pred_np[(m0_np == 0) & (m1_np == 0) & (m2_np == 0)] = 0
            output_img = sitk.GetImageFromArray(pred_np)
            output_img.CopyInformation(sitk.ReadImage(in_path_1))
            sitk.WriteImage(output_img, out_path)


if __name__ == '__main__':
    val_path = r'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData_missT2w'
    save_path = r'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData_missT2w_Syn'
    os.makedirs(save_path, exist_ok=True)

    val_path_list = os.listdir(val_path)
    val_path_list.sort()
    for val in tqdm.tqdm(val_path_list):
        test_one_case(os.path.join(val_path, val), os.path.join(save_path, val))

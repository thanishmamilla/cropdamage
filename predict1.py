# # coding=utf-8
# import os
# import torch.utils.data
# from torch.utils.data import DataLoader
# from data_utils import TestDatasetFromFolder
# from tqdm import tqdm
# import argparse
# import numpy as np
# from model.network import CDNet
# from PIL import Image

# parser = argparse.ArgumentParser(description='Test Change Detection Models')
# parser.add_argument('--gpu_id', default="0,1,2,3", type=str, help='which gpu to run.')
# parser.add_argument('--model_dir', default='netCD_epoch_43.pth', type=str)
# parser.add_argument('--batch_size', default=8, type=int, help='channel of input image')
# parser.add_argument('--crop_size', default=512, type=int, help='channel of input image')
# parser.add_argument('--path_img1', default='Dataset/test/time1', type=str, help='whether used cbam trick')
# parser.add_argument('--path_img2', default='Dataset/test/time2', type=str, help='whether used cbam trick')
# parser.add_argument('--save_dir', default='Output_images/', type=str )

# opt = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if not os.path.exists(opt.save_dir):
#     os.mkdir(opt.save_dir)

# CDNet =  CDNet(img_size=opt.crop_size).to(device, dtype=torch.float)
# CDNet.load_state_dict(torch.load(opt.model_dir))
# CDNet.eval()

# if __name__ == '__main__':
#     test_set = TestDatasetFromFolder(opt, opt.path_img1, opt.path_img2, None)  # Pass None for labels
#     print("Data Loader size: ",len(test_set))
#     test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=opt.batch_size, shuffle=True)
#     test_bar = tqdm(test_loader)

#     for image1, image2, image_name in test_bar:  # No need to load labels
#         image1 = image1.to(device, dtype=torch.float)
#         image2 = image2.to(device, dtype=torch.float)

#         with torch.no_grad():
#             output, _, _ = CDNet(image1, image2)
#             output = torch.argmax(output, 1).squeeze().cpu().numpy()

#         for i in range(len(image_name)):
#             result = Image.fromarray(output[i].astype('uint8'))
#             result.save(opt.save_dir + image_name[i])



# Import necessary libraries
import os
import torch
from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder
from tqdm import tqdm
from model.network import CDNet
from PIL import Image
import numpy as np
# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define arguments for prediction
class Args:
    pass


opt = Args()
opt.gpu_id = "0"  # Specify GPU ID if multiple GPUs are available
opt.model_dir = 'netCD_epoch_43.pth'  # Path to the trained model
opt.batch_size = 8  # Batch size for prediction
opt.crop_size = 512  # Crop size for input images
opt.path_img1 = 'repeatfolder/time1'  # Path to directory containing time 1 images
opt.path_img2 = 'repeatfolder/time2'  # Path to directory containing time 2 images
opt.save_dir = 'Output_images1/'  # Directory to save predicted images
opt.suffix = '.png'  # Suffix of image files

# Create save directory if it doesn't exist
if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

# Load the trained model
CDNet =  CDNet(img_size=opt.crop_size).to(device, dtype=torch.float)
CDNet.load_state_dict(torch.load(opt.model_dir))
CDNet.eval()

# Create test dataset and data loader
test_set = TestDatasetFromFolder(opt, opt.path_img1, opt.path_img2, None)  # Pass None for labels
test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
test_bar = tqdm(test_loader)

# Perform prediction
for image1, image2, image_name in test_bar:  # No need to load labels
    image1 = image1.to(device, dtype=torch.float)
    image2 = image2.to(device, dtype=torch.float)

    with torch.no_grad():
        output, _, _ = CDNet(image1, image2)
        output = torch.argmax(output, 1).squeeze().cpu().numpy()

    # for i in range(len(image_name)):
    #     result = Image.fromarray(output[i].astype('uint8'))
    #     # result.save(os.path.join(opt.save_dir))
    #     result.save(os.path.join(opt.save_dir, f'result{i}.png'))
    for i in range(len(image_name)):
    # Convert pixel values of 1 to 255
        output[i][output[i] == 1] = 255
        
        # Convert the modified array to a PIL Image
        result = Image.fromarray(output[i].astype('uint8'))
        
        # Save the image with a filename that includes the index 'i'
        result.save(os.path.join(opt.save_dir, f'result{i}.png'))


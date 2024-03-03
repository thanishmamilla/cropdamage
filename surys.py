
# # from tqdm import tqdm
# # from PIL import Image

# import os
# import torch
# from torch.utils.data import DataLoader
# from data_utils import TestDatasetFromFolder
# from tqdm import tqdm
# from model.network import CDNet
# from PIL import Image
# import numpy as np
# class Args:
#     pass
# class CDNetPredictor:
#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def predict_images(self, opt):
#         # Create save directory if it doesn't exist
#         if not os.path.exists(opt.save_dir):
#             os.mkdir(opt.save_dir)

#         # Load the trained model
#         CDNet = CDNet(img_size=opt.crop_size).to(self.device, dtype=torch.float)
#         CDNet.load_state_dict(torch.load(opt.model_dir))
#         CDNet.eval()

#         # Create test dataset and data loader
#         test_set = TestDatasetFromFolder(opt, opt.path_img1, opt.path_img2, None)  # Pass None for labels
#         test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
#         test_bar = tqdm(test_loader)

#         # Perform prediction
#         for image1, image2, image_name in test_bar:  # No need to load labels
#             image1 = image1.to(self.device, dtype=torch.float)
#             image2 = image2.to(self.device, dtype=torch.float)

#             with torch.no_grad():
#                 output, _, _ = CDNet(image1, image2)
#                 output = torch.argmax(output, 1).squeeze().cpu().numpy()

#             for i in range(len(image_name)):
#                 # Convert pixel values of 1 to 255
#                 output[i][output[i] == 1] = 255
                
#                 # Convert the modified array to a PIL Image
#                 result = Image.fromarray(output[i].astype('uint8'))
                
#                 # Save the image with a filename that includes the index 'i'
#                 result.save(os.path.join(opt.save_dir, f'result{i}.png'))



import os
import torch
from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder
from tqdm import tqdm
from model.network import CDNet
from PIL import Image
import numpy as np

class Args:
    pass

class CDNetPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_images(self, opt):
        # Create save directory if it doesn't exist
        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)

        # Load the trained model
        cdnet_model = CDNet(img_size=opt.crop_size).to(self.device, dtype=torch.float)
        cdnet_model.load_state_dict(torch.load(opt.model_dir))
        cdnet_model.eval()

        # Create test dataset and data loader
        test_set = TestDatasetFromFolder(opt, opt.path_img1, opt.path_img2, None)  # Pass None for labels
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
        test_bar = tqdm(test_loader)

        # Perform prediction
        for image1, image2, image_name in test_bar:  # No need to load labels
            image1 = image1.to(self.device, dtype=torch.float)
            image2 = image2.to(self.device, dtype=torch.float)

            with torch.no_grad():
                output, _, _ = cdnet_model(image1, image2)
                output = torch.argmax(output, 1).squeeze().cpu().numpy()

            for i in range(len(image_name)):
                # Convert pixel values of 1 to 255
                output[i][output[i] == 1] = 255
                
                # Convert the modified array to a PIL Image
                result = Image.fromarray(output[i].astype('uint8'))
                
                # Save the image with a filename that includes the index 'i'
                result.save(os.path.join(opt.save_dir, f'result{i}.png'))

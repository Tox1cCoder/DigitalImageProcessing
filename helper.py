import os
from bsrgan.utils import utils_image as util
from bsrgan import BSRGAN

class CustomBSRGAN(BSRGAN):
    def __init__(self, model_path, device, hf_model=False, output_dir=None):
        super().__init__(model_path, device, hf_model)
        self.output_dir = output_dir
        self.save = True
    
    def predict(self, img_path):
        img = util.imread_uint(img_path, n_channels=3)
        img = util.uint2tensor4(img)
        img = img.to(self.device)
        img = self.model(img)
        img = util.tensor2uint(img)
        image_name = os.path.basename(img_path)

        save_path = os.path.join(self.output_dir, "enhanced_" + image_name)
        
        util.mkdir(os.path.dirname(save_path))
        result = util.imsave(img, save_path)
        return result
import json
import torch
from torch.utils.data import Dataset, DataLoader

from commons import get_model, transform_image


class Inference(Dataset):
    def __init__(self):
        """
        Args:
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_base_model()
    
    def get_base_model(self):
        self.model, self.classes = get_model(model_type="vgg19_fine_tunned")
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, input_tensor):
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output
    
    def predict_class(self, image):
      
        tensor = transform_image(image=image)
        output = self.predict(tensor)
        _, predicted = torch.max(output, 1)
        predicted_idx = str(predicted.item())
        return predicted_idx,self.classes[predicted_idx]

    
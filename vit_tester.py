#!/usr/bin/python

import os
import numpy as np
from hi_dim_tester import HiDimTester
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTForImageClassification
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from functools import partial


class ViTTester(HiDimTester):
    """
    Concrete PoC implementation of tester for Vision Transformer on square images of typical resolution (i.e., more than 32x32,
    but probably less than 500x500 for efficiency's sake)
    """

    def __init__(self, radius, img_side, idx, sampling_fraction=1):
        self.inner_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.to_img_transform = transforms.ToPILImage()
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        self.img_idx = idx
        self.image_side = img_side
        model = self.ViTLogitPredict()
        super().__init__(model, radius, img_side**2, sampling_fraction)

    def vector_to_gray_image(self, vector, xdim, ydim, scale_factor=1/255):
        """
        Converts 1D vector to 2D grayscale image of given dimensions (assumes reshape possible).

        Args:
        vector: given 1D vector
        xdim: desired number of pixels in x-dim
        ydim: desired number of pixels in y-dim
        scale_factor: divide each vector value by this for proper grayscale output
        
        Returns:
        grayscale image object of given dimensions
        """
        
        x = vector.reshape(1,xdim,ydim)*scale_factor
        y = x.expand(3,xdim,ydim)
        return self.to_img_transform(y)

    def gray_image_to_vector(self, image):
        tensor = self.transform(image).float()
        xdim = tensor.shape[1]
        ydim = tensor.shape[2]
        return tensor[0][:][:].reshape(xdim*ydim)

    def image_to_gray_image(self, image):
        gray_image = ImageOps.grayscale(image)
        gray_img_tensor = self.transform(gray_image).float()/255
        xdim = gray_img_tensor.shape[1]
        ydim = gray_img_tensor.shape[2]
        new_gray_img_tensor = gray_img_tensor.expand(3,xdim,ydim)
        return self.to_img_transform(new_gray_img_tensor)

    def ViT_predict(self, image):
        """
        Processes image through Vision Transformer to give predicted label

        Args:
        image: image object

        Returns:
        tuple of predicted label and corresponding index 
        """        
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.inner_model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        return self.inner_model.config.id2label[predicted_label], predicted_label


    def ViT_predict_logit_and_avg(self, image, idx):
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = self.inner_model(**inputs).logits

        return logits[0][idx], torch.mean(logits[0]).item()
    
    def ViT_predict_logit(self, image, idx):
        """
        Gives predicted logit value of the image at given index value

        Args:
        image: image object
        idx: desired index

        Returns:
        ResNet predicted logit value (float) at index position for this image
        """
        
        inputs = self.processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = self.inner_model(**inputs).logits

        return logits[0][idx]

    def vit_idx_logit(self, idx, dim, point):
        return self.ViT_predict_logit(self.vector_to_gray_image(point, dim, dim), idx)
        
    def ViTLogitPredict(self):
        return partial(self.vit_idx_logit, self.img_idx, self.image_side)


def main():


    # testing cat image

    image_side = 100
    n_dim = image_side**2
    mag = 100
    sampling_fraction = 0.001

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/gallina/'
    #outfile = open('gallina.vit.out', 'w')

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/cane/'
    #outfile = open('cane.vit.out', 'w')


    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/cavallo/'
    #outfile = open('cavallo.vit.out', 'w')

    
    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/elefante/'
    #outfile = open('elefante.vit.out', 'w') 

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/pecora/'
    #outfile = open('pecora.vit.out', 'w') 

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/scoiattolo/'
    #outfile = open('scoiattolo.vit.out', 'w')
    #outfile = open('scoiattolo.1.out', 'w') 

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/farfalla/'
    #outfile = open('farfalla.vit.out', 'w')

    path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/ragno/'
    outfile = open('ragno.vit.out', 'w')

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/mucca/'
    #outfile = open('mucca.vit.out', 'w')

    #path = '/Users/lordkersting/neuro/Downloads/animals_small/raw-img/gatto/'
    #outfile = open('gatto.vit.out', 'w')

    image_path = "images/"
    num_steps = 25
    count = 0
    for file in os.listdir(path):
        count += 1
        if count > 100:
            break
        image = Image.open(path+file)
        idx = 285  # Egpytian Cat
        small_image = transforms.Resize((image_side,image_side)).forward(image)


        currtester = ViTTester(mag, image_side, idx, sampling_fraction)
        small_gray_image = currtester.image_to_gray_image(small_image)
        curr_point = currtester.gray_image_to_vector(small_gray_image)
        img_class, img_idx = currtester.ViT_predict(small_gray_image)
        #print(f"Starting out with central image: {img_idx}, {img_class}")
        #print(f"Anharmoniticity for {idx} is {currtester.anharmoniticity(curr_point)}")

        orig_anharm = currtester.anharmoniticity(curr_point)
        pred_logit, avg_logit = currtester.ViT_predict_logit_and_avg(small_gray_image, img_idx)
        
        for _ in range(num_steps):
            curr_point, anharm = currtester.follow_anharmonic_gradient(curr_point)


        curr_image = currtester.vector_to_gray_image(curr_point, 100, 100)
        final_class, final_idx = currtester.ViT_predict(currtester.vector_to_gray_image(curr_point, image_side, image_side))
        final_pred_logit, final_avg_logit = currtester.ViT_predict_logit_and_avg(curr_image, final_idx)
        output_string = f"{file}\t{img_idx}\t{img_class}\t{pred_logit}\t{avg_logit}\t{orig_anharm}\t{final_idx}\t{final_class}\t{final_pred_logit}\t{final_avg_logit}\t{anharm}\n"
        print(output_string)
        outfile.write(output_string)
        curr_image.save(image_path + f"{file}.{num_steps}.vit.adv.jpg")


    outfile.close()
    
    
if __name__ == "__main__":
    main()
    

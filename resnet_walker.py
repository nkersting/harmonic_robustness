#!/usr/bin/python


from resnet_tester import ResNetTester
import torchvision.transforms as transforms
from PIL import Image, ImageOps

def main():


    # starting with cat image, progressively alter image to greater anharmoniticities

    
    number_of_steps = 100  # traverse this number of steps in direction of anharmoniticity gradient
    image_side = 100
    n_dim = image_side**2
    mag = 100
    sampling_fraction = 0.001
    image = Image.open('../test_images/cat707.jpg')
    idx = 285  # Egpytian Cat
    small_image = transforms.Resize((image_side,image_side)).forward(image)


    currtester = ResNetTester(mag, image_side, idx, sampling_fraction)
    small_gray_image = currtester.image_to_gray_image(small_image)
    curr_point = currtester.gray_image_to_vector(small_gray_image)
    img_class, img_idx = currtester.ResNet_predict(small_gray_image)
    print(f"Starting out with central image: {img_idx}, {img_class}")

    print(f"Anharmoniticity for {idx} is {currtester.anharmoniticity(curr_point)}")

    for i in range(number_of_steps):
        curr_point, anharm = currtester.follow_anharmonic_gradient(curr_point)
        curr_image = currtester.vector_to_gray_image(curr_point, image_side, image_side)
        print(f"{i} Closest adversarial point with anharm={anharm} is {currtester.ResNet_predict(curr_image)}")
        curr_image.save(f"{i}-step.jpg")
    
if __name__ == "__main__":
    main()
    

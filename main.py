import numpy as np
import cv2
from PIL import Image, ImageEnhance
from matplotlib import cm
import matplotlib.pyplot as plt


def one_hot_encode(label):

    """ Функция осуществляет перекодировку текстового "названия" сигнала
     в список элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка
     Выходные параметры: метка ввиде списка

     Пример:
        one_hot_encode("red") должно возвращать:        [1, 0, 0, 0, 0]
        one_hot_encode("yellow") должно возвращать:     [0, 1, 0, 0, 0]
        one_hot_encode("green") должно возвращать:      [0, 0, 1, 0, 0]
        one_hot_encode("yellow_red") должно возвращать: [0, 0, 0, 1, 0]
        one_hot_encode("off") должно возвращать:        [0, 0, 0, 0, 1]

     """
    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0, 0, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1, 0, 0]
    elif label == "yellow_red":
        one_hot_encoded = [0, 0, 0, 1, 0]
    elif label == "off":
        one_hot_encoded = [0, 0, 0, 0, 1]

    return one_hot_encoded


def increase_contrast(img):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    output_img = enhancer.enhance(factor=1.25)
    output_arr = cv2.bitwise_not(np.array(output_img))
    return output_arr


def detect_contours(img):
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    img_canny = cv2.Canny(img, 150, 200)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    img_closed_conts = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    circles = cv2.HoughCircles(img_closed_conts, cv2.HOUGH_GRADIENT, 1, 20, 50, 30)
    for i in circles:
        # draw the outer circle
        print(i)
        cv2.circle(img_copy,(i[0],i[1]),i[2],(0,255,0),2)
    #contours, _ = cv2.findContours(img_closed_conts, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    #img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(img_copy, contours, contourIdx=-1, color=(255, 255, 0), thickness=2)
    cv2.imshow('closed_contours', img_copy)
    while cv2.waitKey(1) != ord('n'):
        continue
    #return contours

    
# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):

    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green", "yellow_red", "off"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = cv2.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list


def standardize_input(image):
    """Приведение изображений к стандартному виду. 
    Входные данные: изображение (bgr)
    Выходные данные: стандартизированное изображений.
    """

    image = increase_contrast(image)
    if len(image.shape) >= 3:
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        val = cv2.resize(hsv_img[:, :, 2], (30, 90))
    else:
        val = image
    #val = hsv_img[:, :, 2]
    cutoff_x, cutoff_y = 6, 0
    val = val[cutoff_x:90 - cutoff_x, cutoff_y:30 - cutoff_y]

    contours = detect_contours(val)

    val = cv2.GaussianBlur(val, (5, 5), sigmaX=6.0)

    #cv2.imshow('frame', val)
    #while cv2.waitKey(1) != ord('q'):
    #    continue

    ## TODO: Если вы хотите преобразовать изображение в формат, одинаковый для всех изображений, сделайте это здесь.
    return val


def predict_label(rgb_image):
    """
     функция определения сигнала светофора по входному изображению

     Входные данные: изображение (bgr)
     Выходные данные: метка в формате списка (смотри one_hot_encode)

    """

    #img = cv2.imread(file_path)
    #print(rgb_image.shape)
    #cv2.imshow('wth', rgb_image)
    #while True:
    #    if cv2.waitKey(1) == ord('q'):
    #        break
    val = standardize_input(rgb_image)
    
    #cv2.imshow('val', val)
    h, w = val.shape
    light_h, light_w = h // 3, w
    red_sum = np.sum(val[:light_h, :])
    yellow_sum = np.sum(val[light_h + 1:2 * light_h, :])
    green_sum = np.sum(val[2 * light_h + 1:, :])
    #color_sums.append([red_sum, yellow_sum, green_sum])
    #print(red_sum, yellow_sum, green_sum)

    dark_thr = 27000    # image darkness threshold
    difference_thr = 19000   # max color difference

    result = ''

    if abs(red_sum - yellow_sum) < difference_thr and abs(yellow_sum - green_sum) < difference_thr and \
        abs(red_sum - green_sum) < difference_thr:
        result = 'off'
        #print('hi')
    else:
        if red_sum > dark_thr and yellow_sum > dark_thr and green_sum > dark_thr:
            if red_sum > yellow_sum and red_sum > green_sum:
                if abs(red_sum - yellow_sum) < difference_thr and yellow_sum > green_sum:
                    result = 'yellow_red'
                else:
                    result = 'red'
            elif yellow_sum > red_sum and yellow_sum > green_sum:
                if abs(red_sum - yellow_sum) < difference_thr and red_sum > green_sum:
                    result = 'yellow_red'
                else:
                    result = 'yellow'
            elif green_sum > red_sum and green_sum > yellow_sum:
                result = 'green'
        else:
            result = 'off'
            #print('hi')
    
    encoded_label = one_hot_encode(result)  # по умолчанию, говорит что на всех изображения жёлтый сигнал

    ## TODO: ваша функция распознавания сигнала светофора должна быть здесь.
    return encoded_label

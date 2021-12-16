# -*- coding: utf-8 -*-
# Файл служит для определения точности вашего алгоритма
# и не предназначен для редактирования участниками олимпиады
# Для получения оценки точности, запустите файл на исполнение

import cv2  # computer vision library
import helpers  # helper functions
import random
import main


# Image data directories
def load_data():
    IMAGE_DIR_TRAINING = "traffic_light_images/training/"
    #IMAGE_DIR_VALIDATION = "traffic_light_images/val/"
    TRAINING_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)
    #VALIDATION_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_VALIDATION)
    #return TRAINING_IMAGE_LIST, VALIDATION_IMAGE_LIST
    return TRAINING_IMAGE_LIST, []


# приведение всего набора изображений к стандартному виду
def standardize(image_list):
    """Функция осуществляет приведение всего набора изображений к стндартному виду, а метки перводит в формат списка
    Входные данные: список изображений
    Выходные данные: стандартизированный список [изображение, метка]
    """

    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the image
        standardized_im = eval.standardize_input(image)

        # One-hot encode the label
        one_hot_label = eval.one_hot_encode(label)

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))

    return standard_list


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    """Определение точности работа алгоритма
    Сравниваются результаты классификации вашего алгоритма и истинныме метки

    Входные данные: массив с тестовыми изображениями и метками к ним
    Выходные данные: массив с неправильно классифицированными метками

    Этот код используется для тестирования и не должен изменяться
    """

    misclassified_images_labels = []

    for image in test_images:

        im = image[0]
        true_label = image[1]
        assert (len(true_label) == 5), "Метка имеет не верную длинну (нужно 5 значений)."

        predicted_label = eval.predict_label(im)
        assert (len(predicted_label) == 5), "Метка имеет не верную длинну (неужно 5 значений)."

        if predicted_label != true_label:
            misclassified_images_labels.append((im, predicted_label, true_label))

    return misclassified_images_labels


def main():
    TRAIN_IMAGE_LIST, VALIDATION_IMAGE_LIST = load_data()
    # Standardize the test data
    STANDARDIZED_TRAIN_LIST = standardize(TRAIN_IMAGE_LIST)
    #STANDARDIZED_VAL_LIST = standardize(VALIDATION_IMAGE_LIST)

    # Shuffle the standardized test data
    random.shuffle(STANDARDIZED_TRAIN_LIST)
    #random.shuffle(STANDARDIZED_VAL_LIST)

    # Find all misclassified images in a given test set
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TRAIN_LIST)

    # Accuracy calculations
    total = len(STANDARDIZED_TRAIN_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Accuracy: ' + str(accuracy))
    print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))
    print(cv2.useOptimized())
    

def show_data():
    data, _ = load_data()
    data = standardize(data)
    n = 0
    print(data[0])
    while cv2.waitKey(1) != ord('q'):
        cv2.imshow(str(n), data[n][0])
        if cv2.waitKey(1) == ord('n'):
            n += 1

if __name__ == '__main__':
    main()

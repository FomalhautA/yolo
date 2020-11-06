import cv2


def visualize(img_path, predict, ground_truth=None, class_names=('Car', 'Truck', 'Pedestrian')):
    """

    :param img_path:
    :param predict:
    :param ground_truth:
    :param class_names:
    :return:
    """
    xmin, ymin = (100, 30)
    xmax, ymax = (130, 100)
    offset = 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.imread(img_path)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=3)
    cv2.imshow('Image', img)
    cv2.putText(img, 'Car', (xmin, ymin - offset), font, fontScale=1.2, color=(0, 255, 0), thickness=2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

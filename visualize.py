import cv2
import os
import numpy as np

import matplotlib.pyplot as plt


def plot_pr_rr(rr, pr):
    """

    :param rr: list of recall rate
    :param pr: list of precision rate
    :return:
    """
    plt.figure()
    plt.plot(rr, pr, 'b')
    plt.show()


def plot_APs(models, mAPs, class_aps, reverse_classes):
    """

    :param models: list, model ids
    :param mAPs: list of mAPs, same shape with models
    :param class_aps: with shape, models * 3
    :param reverse_classes: dict, id as key, class name as values
    :return:
    """
    plt.figure()
    plt.title('APs')
    plt.plot(models, mAPs, label='mAP')
    for i in range(len(reverse_classes)):
        plt.plot(models, [item[i] for item in class_aps], label='AP for {}'.format(reverse_classes[i]))

    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    # plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig(os.path.join('./performance', 'APs.jpg'))
    plt.show()


def plot_mean_pr_rr_curves(models, pr_rr_curves):
    """

    :param models: list, model ids
    :param pr_rr_curves: nd list, with shape: models * num_classes * bins * 2
    :return:
    """
    plt.figure()
    pr_rr_mean = np.mean(pr_rr_curves, axis=1)  # models * bins * 2
    for i in range(len(models)):
        plt.plot([item[0] for item in pr_rr_mean[i]], [item[1] for item in pr_rr_mean[i]], label=models[i])
    plt.title('mean PR-RR curve')
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.legend(loc='lower left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig(os.path.join('./performance', 'mean_pr_rr_curve.jpg'))
    plt.show()


def plot_pr_rr_curves(models, pr_rr_curves, reverse_classes):
    """

    :param models: list, model ids
    :param pr_rr_curves: nd array, with shape: models * num_class * bins * 2
    :param reverse_classes: dict, id as key, class name as values
    :return:
    """
    for i in range(len(reverse_classes.keys())):
        plt.figure()
        for j in range(len(models)):
            pr_rrs = pr_rr_curves[j][i]
            plt.plot([item[0] for item in pr_rrs], [item[1] for item in pr_rrs], label=models[j])
        plt.xlabel('Recall Rate')
        plt.ylabel('Precision Rate')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.title('PR-RR curves for {}'.format(reverse_classes[i]))
        plt.legend(loc='lower left')
        plt.savefig(os.path.join('./performance', 'pr_rr_curve_{}.jpg'.format(reverse_classes[i])))
        plt.show()


def visualize_evaluation(models, pr_rr_curves, mAPs, class_aps, reverse_classes):
    """

    :param models: list, model ids
    :param pr_rr_curves: nd list, with shape: models * num_classes * bins * 2
    :param mAPs: list of mAPs, same shape with models
    :param class_aps:
    :param reverse_classes: dict, id as key, class name as values
    :return:
    """
    plot_APs(models, mAPs, class_aps, reverse_classes)

    plot_mean_pr_rr_curves(models, pr_rr_curves)

    plot_pr_rr_curves(models, pr_rr_curves, reverse_classes)


def show_predicted_img(fname, labels, predicts, data_folder):
    """

    :param fname:
    :param labels:
    :param predicts:
    :param data_folder:
    :return:
    """
    image = cv2.imread(os.path.join(data_folder, fname))
    for label in labels:
        xmin, ymin, xmax, ymax, fname, classname, url = label
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        cv2.putText(image, classname, org=(xmin, ymin - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, thickness=1, color=(0, 255, 0))

    for pred in predicts:
        xmin, ymin, xmax, ymax, fname, classname, url = pred
        top_left = (xmin, ymin)
        bottom_right = (xmax, ymax)
        cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=2)
        cv2.putText(image, classname, org=(xmin, ymin - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, thickness=1, color=(0, 0, 255))

    cv2.namedWindow('current_image', cv2.WINDOW_AUTOSIZE)

    cv2.imshow('current_image', image)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()

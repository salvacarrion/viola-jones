import time
import glob
import numpy as np
from PIL import Image, ImageDraw
from scipy.misc import toimage
import matplotlib.pyplot as plt

from features import RectangleRegion, HaarFeature
from progress.bar import Bar
import multiprocessing


def imshow(img):
    toimage(img).show()


def load_image(image_path, as_numpy=False):
    pil_img = Image.open(image_path)
    if as_numpy:
        return np.array(pil_img)
    else:
        return pil_img


def rgb2gray(img):
    # Formula: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def integral_image(img):
    """
    Optimized version of Summed-area table
    ii(-1, y) = 0
    s(x, -1) = 0
    s(x, y) = s(x, y-1) + i(x, y)  # Sum of column X at level Y
    ii(x, y) = ii(x-1, y) + s(x, y)  # II at (X-1,Y) + Column X at Y
    """
    h, w = img.shape

    s = np.zeros(img.shape, dtype=np.uint32)
    ii = np.zeros(img.shape, dtype=np.uint32)

    for x in range(0, w):
        for y in range(0, h):
            s[y][x] = s[y - 1][x] + img[y][x] if y - 1 >= 0 else img[y][x]
            ii[y][x] = ii[y][x - 1] + s[y][x] if x - 1 >= 0 else s[y][x]
    return ii


def integral_image_pow2(img):
    """
    Squared version of II
    """
    return integral_image(img**2)


def build_features(img_w, img_h, shift=1, scale_factor=1.25, min_w=4, min_h=4):
    """
    Generate values from Haar features

    White rectangles substract from black ones
    """
    features = []  # [Tuple(positive regions, negative regions),...]

    # Scale feature window
    for w_width in range(min_w, img_w + 1):
        for w_height in range(min_h, img_h + 1):

            # Walk through all the image
            x = 0
            while x + w_width < img_w:
                y = 0
                while y + w_height < img_h:

                    # Possible Haar regions
                    immediate = RectangleRegion(x, y, w_width, w_height)  # |X|
                    right = RectangleRegion(x + w_width, y, w_width, w_height)  # | |X|
                    right_2 = RectangleRegion(x + w_width * 2, y, w_width, w_height)  # | | |X|
                    bottom = RectangleRegion(x, y + w_height, w_width, w_height)  # | |/|X|
                    # bottom_2 = RectangleRegion(x, y + w_height * 2, w_width, w_height)  # | |/| |/|X|
                    bottom_right = RectangleRegion(x + w_width, y + w_height, w_width, w_height)  # | |/| |X|

                    # [Haar] 2 rectagles *********
                    # Horizontal (w-b)
                    if x + w_width * 2 < img_w:
                        features.append(HaarFeature([immediate], [right]))
                    # Vertical (w-b)
                    if y + w_height * 2 < img_h:
                        features.append(HaarFeature([bottom], [immediate]))

                    # [Haar] 3 rectagles *********
                    # Horizontal (w-b-w)
                    if x + w_width * 3 < img_w:
                        features.append(HaarFeature([immediate, right_2], [right]))
                    # # Vertical (w-b-w)
                    # if y + w_height * 2 < img_h:
                    #     features.append(HaarFeature([immediate, bottom_2], [bottom]))

                    # [Haar] 4 rectagles *********
                    if x + w_width * 2 < img_w and y + w_height * 2 < img_h:
                        features.append(HaarFeature([immediate, bottom_right], [bottom, right]))

                    y += shift
                x += shift
    return features  # np.array(features)


def apply_features(X_ii, features):
    """
    Apply build features (regions) to all the training data (integral images)
    """

    X = np.zeros((len(features), len(X_ii)), dtype=np.int32)
    # 'y' will be kept as it is => f0=([...], y); f1=([...], y),...

    bar = Bar('Processing features', max=len(features), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
    for j, feature in bar.iter(enumerate(features)):
    # for j, feature in enumerate(features):
    #     if (j + 1) % 1000 == 0 and j != 0:
    #         print("Applying features... ({}/{})".format(j + 1, len(features)))

        # Compute the value of feature 'j' for each image in the training set (Input of the classifier_j)
        X[j] = list(map(lambda ii: feature.compute_value(ii), X_ii))
    bar.finish()

    return X


def show_sample(x, y, y_pred):
    target = "Face" if y == 1 else "No face"
    pred = "Face" if y_pred == 1 else "No face"
    img_text = "Class: {}  - Prediction: {}".format(target, pred)
    print(img_text)

    plt.title(img_text)
    plt.imshow(x, cmap='gray')
    plt.show()


def evaluate(clf, X, y, show_samples=False):
    metrics = {}
    true_positive, true_negative = 0, 0  # Correct
    false_positive, false_negative = 0, 0  # Incorrect

    for i in range(len(y)):
        prediction = clf.classify(X[i])
        if prediction == y[i]:  # Correct
            if prediction == 1:  # Face
                true_positive += 1
            else:  # No-face
                true_negative += 1
        else:  # Incorrect
            #if show_samples: show_sample(X[i], y[i], prediction)

            if prediction == 1:  # Face
                false_positive += 1
            else:  # No-face
                false_negative += 1

    # Compute metrics
    metrics['true_positive'] = true_positive
    metrics['true_negative'] = true_negative
    metrics['false_positive'] = false_positive
    metrics['false_negative'] = false_negative

    metrics['accuracy'] = (true_positive + true_negative)/(true_positive+false_negative+true_negative+false_positive)
    metrics['precision'] = true_positive / (true_positive+false_positive)
    metrics['recall'] = true_positive / (true_positive+false_negative)  # or Sensitivity
    metrics['specifity'] = true_negative/(true_negative+false_positive)
    metrics['f1'] = (2.0 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

    return metrics


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_images_from_dir(path, extension="*.*"):
    image_list = []
    for filename in glob.glob(path + '/' + extension):  # assuming gif
        img = Image.open(filename)
        #img = img.convert('L')  # To grayscale
        #img = img.resize((19, 19), Image.ANTIALIAS)  # Resize
        img = np.array(img)
        image_list.append(img)

    image_list = np.stack(image_list, axis=0)
    return image_list


def load_dataset(basepath, pos_filename, neg_filename):
    # Load faces/no faces
    pos_samples = np.load(basepath + '/' + pos_filename)
    neg_samples = np.load(basepath + '/' + neg_filename)
    X = np.concatenate([pos_samples, neg_samples], axis=0)

    # Create labels
    y = np.zeros(len(pos_samples)+len(neg_samples))
    y[:len(pos_samples)] = 1

    return X, y


def dir2file(folder, savefile):
    # Load images
    images = load_images_from_dir(folder, "*.pgm")
    print("{} images loaded".format(len(images)))

    # Save images
    np.save(savefile, images)
    print("Done!")


def get_pretty_time(start_time, end_time=None, s="", divisor=1.0):
    if not end_time:
        end_time = time.time()
    hours, rem = divmod((end_time - start_time)/divisor, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{}{:0>2}:{:0>2}:{:05.8f}".format(s, int(hours), int(minutes), seconds)


def draw_bounding_boxes(pil_image, regions, color="green", thickness=3):
    # Prepare image
    source_img = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for rect in regions:
        draw.rectangle(tuple(rect), outline=color, width=thickness)
    return source_img


def non_maximum_supression(regions, threshold=0.5):
    # Code from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    boxes = np.array(regions)
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious


def normalize_image(image):
    ii = integral_image(image)
    mean = np.mean(image)
    stdev = np.std(image)
    norm_img = (image-mean)/stdev
    return norm_img


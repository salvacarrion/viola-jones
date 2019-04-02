import pickle
import time
import numpy as np
from utils import *

from sklearn.feature_selection import SelectPercentile, f_classif
from adaboost import AdaBoost


class ViolaJones:

    def __init__(self, layers, features_path=None):
        assert isinstance(layers, list)
        self.layers = layers  # list with the number T of weak classifiers
        self.clfs = []
        self.base_width, self.base_height = 19, 19  # Size of the images from training dataset
        self.base_scale, self.shift = 1.25, 2
        self.features_path = features_path  # Path to save the features

    def train(self, X, y):
        """
        We train N Viola-Jones classifiers (AdaBoost), each more complex than the previous ones.
        After the first one, each classifier is trained with the positive examples plus
        the false positives of the previous one.
        """

        print("Preparing data...")

        # Prepare training data
        pos_num = np.sum(y)
        neg_num = len(y) - pos_num
        img_h, img_w = X[0].shape  # All training images must have the same size

        # Split positives and negatives samples
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        # Show data info
        print("Summary input data:")
        print("\t- Total faces: {:,} ({:.2f}%)".format(int(pos_num), 100.0 * pos_num / (pos_num + neg_num)))
        print("\t- Total non-faces: {:,} ({:.2f}%)".format(int(neg_num), 100.0 * neg_num / (pos_num + neg_num)))
        print("\t- Total samples: {:,}".format(int(pos_num + neg_num)))
        print("\t- Size (WxH): {}x{}".format(img_w, img_h))

        # Initialize weights and compute integral images
        print("Generating integral images...")
        start_time = time.time()
        X_ii = np.array(list(map(lambda x: integral_image(x), X)), dtype=np.uint32)
        print("\t- Num. integral images: {:,}".format(len(X_ii)))
        print("\t- II/s: " + get_pretty_time(start_time, divisor=len(y)))
        print("\t- Total time: " + get_pretty_time(start_time))

        # Create and apply features
        print("Building features...")
        start_time = time.time()
        features = build_features(img_w, img_h)  # Same features for all images
        print("\t- Num. features: {:,}".format(len(features)))
        print("\t- F/s: " + get_pretty_time(start_time, divisor=len(features)))
        print("\t- Total time: " + get_pretty_time(start_time))

        print("Applying features...")
        start_time = time.time()
        X_f = self.__load_feature_dataset()  # Load feature dataset (if exists)
        if X_f is None:
            X_f = apply_features(X_ii, features)

            if self.features_path:  # Save features
                np.save(self.features_path + "xf" + ".npy", X_f)
                print("Applied features file saved!")
        print("\t- Num. features applied: {:,}".format(len(X_f) * len(features)))
        print("\t- FA/s: " + get_pretty_time(start_time, divisor=len(X_f) * len(features)))
        print("\t- Total time: " + get_pretty_time(start_time))

        # # Percentile optimization
        # indices = SelectPercentile(f_classif, percentile=10).fit(X_f.T, y).get_support(indices=True)
        # X_f = X_f[indices]
        # features = np.array(features)[indices]

        # Train cascade of Viola-Jones classifiers (AdaBoost)
        for i, t in enumerate(self.layers):
            print("[CascadeClassifier] Training {} of out {} layers".format(i+1, len(self.layers)))
            if len(neg_indices) == 0:
                print('Early stop. All samples were correctly classify.')
                break

            # Merge indices and shuffle
            tr_idxs = np.concatenate([pos_indices, neg_indices])
            np.random.shuffle(tr_idxs)

            # Train Viola-Jones (AdaBoost)
            clf = AdaBoost(n_estimators=t)
            clf.train(X_f[:, tr_idxs], y[tr_idxs], features, X_ii[tr_idxs])
            self.clfs.append(clf)

            # Find which non-faces where label as a face
            false_positives = []
            for neg_idx in neg_indices:
                if self.classify(X[neg_idx]) == 1:
                    false_positives.append(neg_idx)
            neg_indices = np.array(false_positives)

    def classify(self, image, scale=1.0):
        """
        If a no-face is found, reject now. Else, keep looking.
        """
        return self.classify_ii(integral_image(image), scale)

    def classify_ii(self, ii, scale=1.0):
        """
        If a no-face is found, reject now. Else, keep looking.
        """
        for clf in self.clfs:  # ViolaJones
            if clf.classify(ii, scale) == 0:
                return 0
        return 1

    def find_faces(self, pil_image):
        """
        Receives a PIL image
        """
        w, h, s = (self.base_width, self.base_height, self.base_scale)
        regions = []

        # Preprocess image
        pil_image = pil_image.convert('L')
        image = np.array(pil_image)
        img_h, img_w = image.shape

        # Compute integral image
        ii = integral_image(image)

        # Sliding window
        # Box must be smaller than the image
        counter = 0
        while int(w*s) < img_w and int(h*s) < img_h:

            # The box must slide just in the image
            for y1 in np.arange(0, int(img_h)-int(h*s), self.shift):
                for x1 in np.arange(0, int(img_w)-int(w*s), self.shift):
                    y1, x1 = int(y1), int(x1)
                    y2, x2 = y1 + int(h*s), x1 + int(w*s)
                    cropped_img = ii[y1:y2, x1:x2]

                    if self.classify_ii(cropped_img, scale=s):  # CascadeClassifier
                        regions.append((x1, y1, x2, y2))

                    counter += 1
                    print("Crops analized: {}".format(counter))

            # Increase scale of the window
            w *= s
            h *= s

        return regions

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __load_feature_dataset(self):
        X_f = None
        # Load precomputed features
        try:
            if self.features_path:
                X_f = np.load(self.features_path + "xf" + ".npy")
                print("Precomputed dataset loaded!")
        except FileNotFoundError:
            pass
        return X_f

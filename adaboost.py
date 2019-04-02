import math
from utils import *

from progress.bar import Bar
from weakclassifier import WeakClassifier


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []
        self.clfs = []

    def train(self, X, y, features, X_ii):
        pos_num = np.sum(y)
        neg_num = len(y)-pos_num
        weights = np.zeros(len(y), dtype=np.float32)

        # Initialize weights
        for i in range(len(y)):
            if y[i] == 1:  # Face
                weights[i] = 1.0 / (pos_num * 2.0)
            else:  # No face
                weights[i] = 1.0 / (neg_num * 2.0)

        # Training
        print("Training...")
        start_time = time.time()
        # bar = Bar('Training viola-jones...', max=self.T, suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
        # for t in bar.iter(range(self.T)):
        for t in range(self.n_estimators):
            print("Training %d classifiers out of %d" % (t+1, self.n_estimators))

            # Normalize weights
            w_sum = np.sum(weights)
            if w_sum == 0.0:
                print("[WARNING] EARLY STOP. WEIGHTS ARE ZERO.")
                break
            weights = weights / w_sum  #np.linalg.norm(weights)

            # Train weak classifiers (one per feature)
            print("Training weak classifiers...")
            start_time2 = time.time()
            weak_classifiers = self.train_estimators(X, y, weights, features)
            print("\t- Num. weak classifiers: {:,}".format(len(weak_classifiers)))
            print("\t- WC/s: " + get_pretty_time(start_time2, divisor=len(weak_classifiers)))
            print("\t- Total time: " + get_pretty_time(start_time2))

            # Select classifier with the lowest error
            start_time2 = time.time()
            print("Selecting best weak classifiers...")
            clf, error, incorrectness = self.select_best(weak_classifiers, X, y, weights)
            #clf, error, incorrectness = self.select_best2(weak_classifiers, weights, X_ii, y,)
            print("\t- Num. weak classifiers: {:,}".format(len(weak_classifiers)))
            print("\t- WC/s: " + get_pretty_time(start_time2, divisor=len(weak_classifiers)))
            print("\t- Total time: " + get_pretty_time(start_time2))

            if error <= 0.5:
                # Compute alpha, beta
                beta = error / (1.0 - error)
                alpha = math.log(1.0 / (beta + 1e-18))  # Avoid division by zero

                # Update weights
                weights = np.multiply(weights, beta ** (1 - incorrectness))

                # Save parameters
                self.alphas.append(alpha)
                self.clfs.append(clf)
            else:
                print(error)
                print("WHAT THE FUCK!????")
        # bar.finish()
        print("<== Training")
        print("\t- Num. classifiers: {:,}".format(self.n_estimators))
        print("\t- FA/s: " + get_pretty_time(start_time, divisor=self.n_estimators))
        print("\t- Total time: " + get_pretty_time(start_time))

    def train_estimators(self, X, y, weights, features):
        """
        Find optimal threshold given current weights
        """
        # Precomputation and initializations
        # This is faster than its numpy version
        weak_clfs = []
        total_pos_weights, total_neg_weights = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos_weights += w
            else:
                total_neg_weights += w

        bar = Bar('Training weak classifiers', max=len(X), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
        for i in bar.iter(range(len(X))):
        # for i in range(len(X)):
        #     if (i+1) % 1000 == 0 and i != 0:
        #         print("Training weak classifiers... ({}/{})".format(i + 1, len(X)))

            # Train weak classifier
            clf = WeakClassifier(haar_feature=features[i])  # Index of features
            clf.train(X[i], y, weights, total_pos_weights, total_neg_weights)
            weak_clfs.append(clf)
        bar.finish()

        return weak_clfs

    def select_best(self, weak_clfs, X, y, weights):
        best_clf, min_error, best_accuracy = None, float('inf'), None

        bar = Bar('Selecting best weak classifier', max=len(weak_clfs), suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s')
        i=-1
        for clf in bar.iter(weak_clfs):
            i+=1
        # for i, clf in enumerate(weak_clfs):
        #     if (i+1) % 1000 == 0 and i != 0:
        #         print("Selecting weak classifiers... ({}/{})".format(i+1, len(weak_clfs)))

            # If real==predicted => real - predicted == 0
            # X[0] => List of feature values of the feature F_i of the all imagenes: [2, -6, 4, -7]
            incorrectness = np.abs(clf.classify_f(X[i]) - y)
            error = float(np.sum(np.multiply(incorrectness, weights))) / len(incorrectness)  # Mean error

            if error < min_error:
                best_clf, min_error, best_accuracy = clf, error, incorrectness

        bar.finish()

        return best_clf, min_error, best_accuracy

    def select_best2(self, classifiers, weights, X_ii, y):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for xii_i, yi, w in zip(X_ii, y, weights):
                correctness = abs(clf.classify(xii_i) - yi)
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(X_ii)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, np.array(best_accuracy)

    def classify(self, X, scale=1.0):
        total = sum(list(map(lambda x: x[0] * x[1].classify(X, scale), zip(self.alphas, self.clfs))))  # Weak classifiers
        return 1 if total >= 0.5 * sum(self.alphas) else 0

"""
Viola-Jones Algorithm

"""

import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import *
from violajones import ViolaJones


def check_data(X, y, num2show=10):
    for i in range(num2show):
        idx = random.randint(0, len(y))
        img = X[idx]
        target = "Face" if y[idx] == 1 else "No face"
        img_text = "Index: {}  -  Target: {}  -  Image size: {}x{}".format(idx, target, *img.shape)
        print(img_text)

        plt.title(img_text)
        plt.imshow(img, cmap='gray')
        plt.show()


def data_augmentation(X, y):
    faces_idxs = np.where(y == 1)[0]
    nonfaces_idxs = np.where(y == 0)[0]

    # Horizontal flip
    X_hf = X[faces_idxs, :, ::-1]

    # Stack samples
    X = np.concatenate((X[faces_idxs], X_hf, X[nonfaces_idxs]))
    y = np.zeros(len(X))
    y[:len(faces_idxs)*2] = 1  # Add new targets

    return X, y


def _ensure_neg_pool(neg_pool_path, caltech_path):
    """Build the bootstrap negative pool from Caltech-256 if missing."""
    if neg_pool_path is None:
        return None
    if not os.path.exists(neg_pool_path):
        if caltech_path is None or not os.path.isdir(caltech_path):
            raise FileNotFoundError(
                "Bootstrap pool not found at {} and no valid CALTECH_PATH "
                "provided to build it from.".format(neg_pool_path))
        print("Bootstrap pool not found, building from {}...".format(caltech_path))
        build_bootstrap_negatives(caltech_path, neg_pool_path)
    print("Loading bootstrap pool: {}".format(neg_pool_path))
    pool = np.load(neg_pool_path)
    print("\t- {:,} patches in pool".format(len(pool)))
    return pool


def train(dataset_path, test_size, layers, data_augment=False, layer_recall=0.99,
          neg_pool_path=None, caltech_path=None,
          target_neg_per_stage=3000, neg_sample_budget=100000):
    # Augmented and non-augmented runs land in different folders so caches
    # (x.npy, y.npy, xf_pos.npy) from a previous setup don't get reused
    # silently when the data layout has changed.
    suffix = "_aug" if data_augment else ""
    features_path = r"./weights/{}{}/".format(test_size, suffix)
    try:
        X = np.load(features_path + "x" + ".npy")
        y = np.load(features_path + "y" + ".npy")
        print("Dataset loaded!")

    except FileNotFoundError:
        # LOADING AND PREPROCESSING *************************************************
        print("Loading new training set...")
        os.makedirs(features_path, exist_ok=True)
        X, y = load_cbcl_dataset(dataset_path, "train")

        if data_augment:
            X, y = data_augmentation(X, y)

        # Shuffle data (Not needed with the CascadeClassifier)
        X, y = unison_shuffled_copies(X, y)

        # Subsample for quick smoke runs
        if isinstance(test_size, int):
            X = X[:test_size]
            y = y[:test_size]

        np.save(features_path + "x" + ".npy", X)
        np.save(features_path + "y" + ".npy", y)
        print("New dataset saved!")

    # Bootstrap negative pool (Caltech-256 patches) ******************************
    neg_pool = _ensure_neg_pool(neg_pool_path, caltech_path)

    # TRAINING ******************************************************************
    print("\nTraining Viola-Jones...")
    clf = ViolaJones(layers=layers, features_path=features_path, layer_recall=layer_recall)
    clf.train(X, y, neg_pool=neg_pool,
              target_neg_per_stage=target_neg_per_stage,
              neg_sample_budget=neg_sample_budget)
    print("Training finished!")

    print("\nSaving weights...")
    clf.save(features_path + 'cvj_weights_' + str(int(time.time())))
    print("Weights saved!")

    return clf


def test(clf, dataset_path, split, name=""):
    # Load test set
    print("\nLoading {}...".format(name))
    X, y = load_cbcl_dataset(dataset_path, split)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(clf, X, y, show_samples=False)

    print("Metrics: [{}]".format(name))
    counter = 0
    for k, v in metrics.items():
        counter += 1
        if counter <= 4:
            print("\t- {}: {:,}".format(k, v))
        else:
            print("\t- {}: {:.3f}".format(k, v))


def pick_weights(path=None):
    """Return `path` if given, otherwise the most recent checkpoint under weights/."""
    if path is not None:
        return path
    import glob
    candidates = sorted(glob.glob("weights/**/cvj_weights_*.pkl"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError("No trained weights under weights/. Run MODE='train' first.")
    return candidates[-1]


def evaluate_splits(weight_path, dataset_path):
    """Load a checkpoint and report metrics on the full CBCL train and test splits."""
    weight_path = pick_weights(weight_path)
    print("Using weights: {}".format(weight_path))
    clf = ViolaJones.load(weight_path)
    test(clf, dataset_path, "train", name="Training set")
    test(clf, dataset_path, "test",  name="Test set")


def find_faces(weight_path=None, image_paths=None, output_dir="images/outputs"):
    weight_path = pick_weights(weight_path)
    print("Using weights: {}".format(weight_path))

    if image_paths is None:
        image_paths = ["images/judybats.jpg", "images/people.png"]

    os.makedirs(output_dir, exist_ok=True)
    clf = ViolaJones.load(weight_path)

    for face_path in image_paths:
        print("Detecting on {}".format(face_path))
        pil_img = load_image(face_path)
        regions = clf.find_faces(pil_img)
        print("\t- raw regions: {}".format(len(regions)))

        if regions:
            regions = non_maximum_supression(regions, threshold=0.3)
            print("\t- after NMS:  {}".format(len(regions)))

        drawn_img = draw_bounding_boxes(pil_img, list(regions), thickness=2)
        out_path = os.path.join(output_dir,
                                os.path.splitext(os.path.basename(face_path))[0] + "_detected.png")
        drawn_img.convert("RGB").save(out_path)
        print("\t- saved -> {}".format(out_path))


def draw_features():
    test_name = 100
    MAX_FACES = 1
    video_folder = "videos"
    frames = []
    fig = plt.figure()

    # Load data
    X = np.load("weights/{}/x".format(test_name) + ".npy")
    y = np.load("weights/{}/y".format(test_name) + ".npy")

    X_faces = X[np.where(y == 1)]

    # Load model
    clf = ViolaJones.load("weights/all_fda/cvj_weights_1554209625.pkl")

    for i, np_img in enumerate(X_faces):

        if i < MAX_FACES:
            # For each Adaboost in the Cascade
            for ab in clf.clfs:

                # For each Weak learner
                for wc in ab.clfs:
                    print("New frame: {}".format(len(frames)))

                    drawn_img = draw_haar_feature(np_img,  wc.haar_feature)
                    img = plt.imshow(drawn_img, cmap="gray")
                    frames.append([img])
                    #plt.savefig(video_folder + "/file_%d.png" % i)
                    #plt.show()

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                    repeat_delay=1000)
    ani.save(video_folder + "/{}/video_{}".format(test_name, test_name) + '.mp4', writer='ffmpeg')
    plt.show()


if __name__ == "__main__":
    import argparse

    def _test_size(value):
        return value if value == "all" else int(value)

    parser = argparse.ArgumentParser(description="Viola-Jones face detector")
    parser.add_argument("mode", choices=["train", "test", "detect"],
                        help="train: fit cascade | test: evaluate metrics | detect: run inference")

    # Common
    parser.add_argument("--dataset-path", default="/Users/salvacarrion/Desktop/mitcbcl",
                        help="Path to the CBCL dataset root")
    parser.add_argument("--caltech-path", default="/Users/salvacarrion/Desktop/256_ObjectCategories",
                        help="Path to Caltech-256 for building bootstrap negatives")
    parser.add_argument("--neg-pool-path", default="/Users/salvacarrion/Desktop/mitcbcl/bootstrap_negatives_19x19g.npy",
                        help="Path to prebuilt bootstrap negative pool (.npy)")

    # Training
    parser.add_argument("--test-size", type=_test_size, default="all",
                        metavar="N|all",
                        help="Subsample training set to N samples, or 'all' for full split (default: all)")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 10, 50, 100],
                        metavar="T",
                        help="Weak learners per cascade stage (default: 1 10 50 100)")
    parser.add_argument("--data-augment", action="store_true", default=True,
                        help="Mirror faces horizontally to double positive class (default: on)")
    parser.add_argument("--no-data-augment", dest="data_augment", action="store_false",
                        help="Disable horizontal-flip augmentation")
    parser.add_argument("--layer-recall", type=float, default=0.99,
                        help="Per-stage face-recall target (default: 0.99)")
    parser.add_argument("--target-neg-per-stage", type=int, default=3000,
                        help="Negatives mined from pool per cascade stage (default: 3000)")
    parser.add_argument("--neg-sample-budget", type=int, default=100000,
                        help="Max patches sampled per stage when mining (default: 100000)")

    # Eval / detection
    parser.add_argument("--weights-path", default=None,
                        help="Checkpoint to load; omit to auto-pick the most recent one")

    # Detection only
    parser.add_argument("--detect-images", nargs="+",
                        default=["images/people.png", "images/clase.png",
                                 "images/physics.jpg", "images/i1.jpg", "images/judybats.jpg"],
                        metavar="IMG",
                        help="Image paths to run face detection on")
    parser.add_argument("--detect-output", default="images/outputs",
                        help="Directory where annotated output images are saved (default: images/outputs)")

    args = parser.parse_args()

    start_time = time.time()
    print("Starting scripting (MODE={})...".format(args.mode))

    if args.mode == "train":
        train(args.dataset_path, args.test_size, args.layers,
              data_augment=args.data_augment, layer_recall=args.layer_recall,
              neg_pool_path=args.neg_pool_path, caltech_path=args.caltech_path,
              target_neg_per_stage=args.target_neg_per_stage,
              neg_sample_budget=args.neg_sample_budget)
    elif args.mode == "test":
        evaluate_splits(args.weights_path, args.dataset_path)
    elif args.mode == "detect":
        find_faces(weight_path=args.weights_path, image_paths=args.detect_images,
                   output_dir=args.detect_output)

    print("\n" + get_pretty_time(start_time, s="Total time: "))


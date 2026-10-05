# Sample outputs

Detections on the five sample images in `images/`, plus before/after figures on FDDB. Every folder can be regenerated with the command next to it.

| Folder | What | Command |
| :-- | :-- | :-- |
| `best/` | Best model, low-confidence boxes dropped (the README gallery) | `python main.py detect --weights-path weights/24/celeba.pkl --detect-min-score 0.8 --detect-images images/people.png images/clase.png images/i1.jpg images/judybats.jpg --detect-output images/outputs/best` |
| `24_celeba/` | Best model, default settings | `python main.py detect --weights-path weights/24/celeba.pkl --detect-output images/outputs/24_celeba` |
| `19_celeba_cbcl/`, `19_cbcl/` | The two 19×19 models, default settings | same, with `weights/19/celeba_cbcl.pkl` or `weights/19/cbcl.pkl` |
| `opencv_default/` | OpenCV's pretrained `haarcascade_frontalface_default` (cv2) | `python tools/baseline_opencv.py detect --images images/people.png images/clase.png images/physics.jpg images/i1.jpg images/judybats.jpg --cascade default` |
| `fddb_comparison/` | FDDB before/after the inference fixes: ground truth green, ours red, OpenCV blue | `python tools/draw_fddb_comparison.py --weights weights/24/celeba.pkl --names 2002/08/02/big/img_314 2002/08/18/big/img_589 2002/08/17/big/img_935` |

Our boxes are face boxes (forehead to chin) mapped from the cascade's training window with the model's `box_transform`; add `--raw-boxes` to see the raw window. `clase.png` and `physics.jpg` show the main limitation on small faces: below ~35 px wide they fall under the 24×24 window and are missed, and crowds produce large false positives.

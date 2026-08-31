# webcamAI

Real-time object detection on a webcam feed. Draws labelled bounding boxes,
announces objects out loud as they appear, and logs every detection to CSV.

Built with SSD MobileNet V2 (pretrained on COCO) via TensorFlow Hub.

## What it does

- Captures frames from the default webcam with OpenCV
- Runs each frame through SSD MobileNet V2 and keeps detections above 50% confidence
- Overlays bounding boxes and `label: confidence` on the live video
- Speaks the name of each **newly appeared** object via text-to-speech
- Writes a timestamped `detections.csv` on exit

## Why I built it

I wanted to understand what "running a model" actually involves end to end, rather
than calling a hosted vision API and getting a JSON blob back. Doing it locally meant
dealing with the parts an API hides: getting frames off the camera, converting colour
spaces, resizing into the tensor shape the model expects, decoding raw output arrays
into something meaningful, and keeping all of that fast enough to look real-time.

The text-to-speech and CSV logging were there to make it feel like a tool instead of a
demo — something with output you could actually go back and look at.

## Running it

Requires Python 3.8+ and a webcam.

```bash
git clone https://github.com/harrisonkay2112/webcamAI.git
cd webcamAI
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python object_detection.py
```

The model (~20 MB) downloads from TensorFlow Hub on first run and is cached
afterwards. Press `q` in the video window to quit.

## What I learned

**The class IDs a model emits are not the list you think they are.** My original
version indexed a dense 80-element COCO label list with `class_id - 1`. But this model
emits IDs from the *90-entry* COCO map, which has gaps at 12, 26, 29, 30, 45, 66, 68,
69, 71 and 83. Everything after the first gap was silently mislabelled — a stop sign
(ID 13) was reported as a parking meter. Nothing crashed and no error was raised; the
output was just quietly wrong, which is the worst failure mode there is. Keying a dict
by the actual ID fixed it. The lesson I'd keep: when you glue two components together,
verify the contract between them instead of assuming the obvious mapping.

**Synchronous calls belong outside the capture loop.** Speech ran on every detection in
every frame, and `pyttsx3.runAndWait()` blocks until it finishes talking. At ~30 fps
with a person in frame that meant the loop stalled constantly and repeated "I see a
person" endlessly. Tracking the set of objects seen last frame and only speaking the
new ones fixed both the stutter and the repetition.

**`opencv-python-headless` was in my requirements alongside `opencv-python`.** They
conflict, and the headless build has no GUI — so `cv2.imshow` cannot work. It happened
to run because the non-headless install won, but the file was wrong.

## What I'd change

- **Detection is the bottleneck.** Every frame goes through the full model. Running
  detection every Nth frame and tracking boxes in between would raise the frame rate a
  lot for very little accuracy loss.
- **Speech is still synchronous.** Moving it to a worker thread would remove the
  remaining hitch when a new object enters frame.
- **The TF Hub URL is legacy.** `tfhub.dev` has been folded into Kaggle Models; the URL
  still resolves but should be repointed.
- **No tests.** The label-map bug would have been caught immediately by a single test
  asserting `COCO_LABELS[13] == "stop sign"`.

## License

MIT — see [LICENSE](LICENSE).

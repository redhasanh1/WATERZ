NEW SORA WATERMARK TRAINING
===========================

Put your Sora video frames here!

STEP 1: Extract frames from Sora video
---------------------------------------
Run EXTRACT_FRAMES.bat and give it a Sora video with watermark.
It will extract frames to images/ folder.

STEP 2: Label the watermarks
-----------------------------
Use labelImg or Roboflow to draw boxes around Sora watermarks.
Save labels in labels/ folder (YOLO format).

Or use the auto-labeling tool: LABEL_FRAMES.bat

STEP 3: Train YOLO
------------------
Run TRAIN_NEW_SORA.bat to train on this new dataset.

FOLDER STRUCTURE:
NEW_SORA_TRAINING/
  ├── images/          <- Put .jpg frames here
  ├── labels/          <- Put .txt YOLO labels here
  └── README.txt

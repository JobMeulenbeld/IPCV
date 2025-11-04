# IPCV Final assignment

## Authors
- Job Meulenbeld s3306232
- Xander Küpers s2430347
- Danny Luchtenveld s2885751

## Create a virutal environment
```python -m venv ./venv```

## Install packages
```pip install opencv-python```
```pip install opencv-contrib-python```

## RUN
- run main.py (requires webcam)
- Place your hand in bottom left corner (open hand spread out fingers)
- When the framecount of your detected hand has reached 30 move: UP, DOWN, LEFT, RIGHT
- Moving LEFT or RIGHT will change the augmentation (filter)
- Moving UP or DOWN will change the warp strength
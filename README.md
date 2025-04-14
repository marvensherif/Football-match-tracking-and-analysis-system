# Soccer Video Analysis System

A system for detecting and tracking players, referees, staff, and the ball in soccer match videos. It classifies players into teams, logs presence times, and annotates the video with bounding boxes and labels.

---

## Overview
This system processes soccer videos to:
- Detect objects (players, goalkeepers, ball, referees, staff) using YOLOv8.
- Track objects across frames with DeepSort.
- Classify players into teams using K-Means clustering on jersey colors.
- Generate an annotated video and log presence times of detected objects.

---

## Features
- **Object Detection**: YOLOv8 detects 8 classes:
  - Players, goalkeepers, ball, main referee, side referees, and staff.
- **Team Classification**: Players are grouped into left/right teams based on jersey colors.
- **Tracking**: DeepSort maintains consistent IDs across frames.
- **Annotation**: Bounding boxes with labels (e.g., `Player-L`, `GK-R`) and tracking IDs.
- **Logging**: Saves timestamps and images of detected objects.

## installations
- use the google collab notebook
- update the main code with the main code in this repo also for requiremnts to avoid errors

## uses
- add your input video at test_videos folder
- you output will be added at output folder
- also players logs folder will be created
-  at line 143,144 at main.py yoy can update max_age=200 to Keep tracks alive longer for occlusions and n_init=5 Require more confirmations to start tracks

## Workflow
- Input Video: Reads frames sequentially.

## Object Detection:

- YOLOv8 detects objects in each frame.

- Filters players for team classification.

- Team Identification (First Frame):

- Extracts grass color to mask non-player regions.

- Uses K-Means to cluster player jersey colors into two teams.

## Tracking:

- DeepSort assigns unique IDs to detected objects.

- Tracks objects across frames with max_age=500 and n_init=5.

## Annotation:

- Draws bounding boxes with team labels (Player-L, GK-R).

- Labels referees, staff, and the ball.

## Logging:

- Records start/end times for each tracked object.

- Saves cropped images of detected objects.

## Functions
- get_grass_color(img)
- Purpose: Identifies dominant grass color in the frame.

- Logic: Uses HSV masking to isolate green regions and compute average color.

- get_players_boxes(result)
- Purpose: Extracts player bounding boxes and images from YOLO results.

- Output: Cropped player images and bounding boxes.

- get_kits_colors(players, grass_hsv, frame)
- Purpose: Extracts jersey colors while ignoring grass background.

- Logic: Masks grass regions and focuses on the upper half of players.

- get_kits_classifier(kits_colors)
- Purpose: Trains a K-Means model to classify players into two teams.

- classify_kits(kits_classifier, kits_colors)
- Purpose: Predicts team labels (0 or 1) for players.

- get_left_team_label(players_boxes, kits_colors, kits_clf)
- Purpose: Determines which team is on the left side of the screen.

- annotate_video(video_path, model)
- Purpose: Main function for video processing, tracking, and annotation.


    

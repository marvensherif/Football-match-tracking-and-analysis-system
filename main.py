import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans
from deep_sort_realtime.deepsort_tracker import DeepSort

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image.
    """
    if img is None or img.size == 0:
        return (0, 0, 0)  # Default color if the image is empty

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    """
    Extracts player bounding boxes and images from YOLO results.
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        label = int(box.cls.tolist()[0])
        if label == 0:  # Only process players
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            player_img = result.orig_img[y1: y2, x1: x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Extracts the dominant kit color for each player.
    """
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        if player_img is None or player_img.size == 0:
            continue

        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)

    return kits_colors

def get_kits_classifier(kits_colors):
    """
    Creates a K-Means classifier to classify players into two teams.
    """
    if len(kits_colors) == 0:
        print("No kit colors found. Skipping KMeans clustering.")
        return None

    # Ensure kits_colors is a 2D array
    kits_colors = np.array(kits_colors).reshape(-1, 3)  # Reshape to (n_samples, 3)

    kits_kmeans = KMeans(n_clusters=2)
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def classify_kits(kits_classifer, kits_colors):
    """
    Classifies a player's kit color into one of two teams.
    """
    if kits_classifer is None or len(kits_colors) == 0:
        return np.array([0])  # Default to team 0 if no classifier or colors

    kits_colors = np.array(kits_colors).reshape(-1, 3)  # Ensure 2D array
    team = kits_classifer.predict(kits_colors)
    return team

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Determines which team is on the left side of the screen.
    """
    left_team_label = 0
    team_0 = []
    team_1 = []

    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].tolist())
        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team == 0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))

    team_0 = np.array(team_0)
    team_1 = np.array(team_1)

    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1

    return left_team_label

def annotate_video(video_path, model):
    """
    Processes the video, detects objects, and annotates the frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Get video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_name = video_path.split('/')[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('./output/'+video_name.split('.')[0] + "_out.mp4",
                                   fourcc,
                                   fps,
                                   (width, height))

    kits_clf = None
    left_team_label = 0
    grass_hsv = None
    # Update tracker initialization (increase max_age, adjust n_init)
    tracker = DeepSort(
    max_age=200,  # Keep tracks alive longer for occlusions
    n_init=5,    # Require more confirmations to start tracks
    embedder="mobilenet",  # Better feature extractor
    max_cosine_distance=0.4,  # Tighter appearance matching
    nn_budget=100
   )

    # Log data
    player_times = {}  # {track_id: [start_time, end_time]}
    player_images = {}
    player_classes = {}  # {track_id: class label}
    # player_teams = {}    # {track_id: image}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        annotated_frame = frame.copy()

        # Run YOLO inference
        result = model(annotated_frame, conf=0.5, verbose=False)[0]

        # Get the players boxes and kit colors
        players_imgs, players_boxes = get_players_boxes(result)
        kits_colors = get_kits_colors(players_imgs, grass_hsv, annotated_frame)

        # Run on the first frame only
        if current_frame_idx == 1:
            kits_clf = get_kits_classifier(kits_colors)
            left_team_label = get_left_team_label(players_boxes, kits_colors, kits_clf)
            grass_color = get_grass_color(result.orig_img)
            grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

        # Convert YOLO detections to DeepSORT format
        detections = []
        for box in result.boxes:
            label = int(box.cls.tolist()[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf.tolist()[0]
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, label))

        # Update DeepSORT tracker with detections
        tracks = tracker.update_tracks(detections, frame=annotated_frame)

        # Annotate frame with tracking IDs and team labels
        for track in tracks:
           # Only draw boxes for tracks updated in THIS frame
            if track.time_since_update > 0:
              continue
            if not track.is_confirmed():
              continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            if x1 >= x2 or y1 >= y2:
              continue

            # Get the original label from YOLO detection
            label = int(track.det_class)

            # If the box contains a player, find to which team he belongs
            if label == 0:
                kit_color = get_kits_colors([annotated_frame[y1:y2, x1:x2]], grass_hsv)
                team = classify_kits(kits_clf, kit_color)
                if team == left_team_label:
                    label = 0  # Player-L
                else:
                    label = 1  # Player-R

            # If the box contains a Goalkeeper, find to which team he belongs
            elif label == 1:
                if x1 < 0.5 * width:
                    label = 2  # GK-L
                else:
                    label = 3  # GK-R

            # Increase the label by 2 because of the two add labels "Player-L", "GK-L"
            else:
                label = label + 2

            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_colors[str(label)], 2)
            cv2.putText(annotated_frame, f"{labels[label]} ID: {track_id}", (x1 - 30, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_colors[str(label)], 2)

            # Update player times and images
            if track_id not in player_times:
                player_times[track_id] = [current_time, current_time]
                player_images[track_id] = annotated_frame[y1:y2, x1:x2]  # Save the player's image
                player_classes[track_id] = labels[label]
            else:
                player_times[track_id][1] = current_time  # Update end time

        # Write the annotated frame to the output video
        output_video.write(annotated_frame)

    # Save player data to a log file
    os.makedirs("player_logs", exist_ok=True)
    # Modify the log writing section
    with open("player_logs/player_log.txt", "w") as log_file:
      for track_id, (start_time, end_time) in player_times.items():
          total_time = end_time - start_time
          player_image = player_images.get(track_id)
          class_info = player_classes.get(track_id, "Unknown")
          # team_info = player_teams.get(track_id, "Unknown")

          if player_image is not None:
              player_image_path = f"player_logs/player_{track_id}.jpg"
              cv2.imwrite(player_image_path, player_image)
              
              log_file.write(f"ID: {track_id}\n")
              log_file.write(f"Class: {class_info}\n")
              # log_file.write(f"Team: {team_info}\n")
              log_file.write(f"Duration: {total_time:.2f}s\n")
              log_file.write(f"Image: {player_image_path}\n")
              log_file.write("-" * 40 + "\n")

    cv2.destroyAllWindows()
    output_video.release()
    cap.release()

if __name__ == "__main__":
    labels = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
    box_colors = {
        "0": (150, 50, 50),
        "1": (37, 47, 150),
        "2": (41, 248, 165),
        "3": (166, 196, 10),
        "4": (155, 62, 157),
        "5": (123, 174, 213),
        "6": (217, 89, 204),
        "7": (22, 11, 15)
    }
    model = YOLO("./weights/last.pt")
    video_path = sys.argv[1]
    annotate_video(video_path, model)
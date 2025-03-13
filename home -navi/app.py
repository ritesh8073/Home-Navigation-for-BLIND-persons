import cv2
import pyttsx3
import torch
import time
import heapq
import numpy as np

# Initialize Text-to-Speech Engine
def speak(instruction, rate=150, volume=1.0):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  # Set speed of speech
    engine.setProperty('volume', volume)  # Set volume level (0.0 to 1.0)
    engine.say(instruction)
    engine.runAndWait()

# Load YOLOv5 Model
def load_model():
    try:
        print("Loading YOLOv5 model...")
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Object Detection Function
def detect_objects(model, frame):
    results = model(frame)  # Detect objects
    detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame of results
    return detected_objects

# A* Algorithm for Pathfinding
def a_star(grid, start, end):
    # Heuristic function (Euclidean distance)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # A* algorithm implementation
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))  # (f_score, g_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, _, current = heapq.heappop(open_list)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] == 0:
                tentative_g_score = g_score.get(current, float('inf')) + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

    return []  # Return empty list if no path found

# Path Detection Function (Canny Edge Detection)
def detect_paths(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Edge detection
    return edges

# Main Function
def main():
    # Load YOLOv5 Model
    model = load_model()
    if not model:
        return

    # Open the Video File
    video_path = "home_video.mp4"  # Ensure the video file path is correct
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    print("Processing video...")
    frame_rate = 30  # You can adjust this based on the video

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the grid size for pathfinding (downsample the image for performance)
    grid_size = 20  # Grid size (each cell represents 20x20 pixels)
    grid_width = width // grid_size
    grid_height = height // grid_size

    all_path_points = []  # To store the entire path points

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Process frames at a specific frame rate to improve performance
        start_time = time.time()

        # Object Detection
        detected_objects = detect_objects(model, frame)

        # Create grid (0 for free space, 1 for obstacles)
        grid = np.zeros((grid_height, grid_width), dtype=int)

        # Mark detected objects as obstacles in the grid
        for _, row in detected_objects.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            grid_xmin, grid_ymin = xmin // grid_size, ymin // grid_size
            grid_xmax, grid_ymax = xmax // grid_size, ymax // grid_size
            grid[grid_ymin:grid_ymax, grid_xmin:grid_xmax] = 1  # Mark as obstacle

        # Start and End points (center of the frame and the bottom of the frame)
        start = (height // 2 // grid_size, width // 2 // grid_size)  # Start point (center of the frame)
        end = (height // grid_size - 1, width // 2 // grid_size)  # End point (center of the bottom)

        # Find path using A* algorithm
        path = a_star(grid, start, end)

        # If path is found, store all points and draw it
        if path:
            all_path_points = path  # Store the entire path

        # Draw the entire path in green (from start to end)
        for (y, x) in all_path_points:
            pixel_x = x * grid_size + grid_size // 2
            pixel_y = y * grid_size + grid_size // 2
            cv2.circle(frame, (pixel_x, pixel_y), 1, (0, 255, 0), -1)

        # Display Object Detection Results
        for _, row in detected_objects.iterrows():
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])), (0, 0, 255), 2)  # Red bounding box
            cv2.putText(frame, row["name"], (int(row['xmin']), int(row['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

            # Calculate distance and direction
            object_center_x = (row['xmin'] + row['xmax']) // 2
            object_center_y = (row['ymin'] + row['ymax']) // 2
            distance = np.linalg.norm(np.array([width // 2, height // 2]) - np.array([object_center_x, object_center_y]))  # Euclidean distance

            # Based on the object position, give navigation commands
            if object_center_x < width // 3:
                speak("Turn left. There is a " + row["name"] + " ahead.")
            elif object_center_x > 2 * width // 3:
                speak("Turn right. There is a " + row["name"] + " ahead.")
            elif object_center_y < height // 3:
                speak("Move forward. " + row["name"] + " detected ahead.")
            elif distance < 100:
                speak(f"Warning! A {row['name']} is very close to you, about {int(distance)} meters away.")
            else:
                speak(f"A {row['name']} is {int(distance)} meters away.")

        # Display the video frame with path and detected objects
        cv2.imshow("Object Detection and Path", frame)

        # Check the time taken to process each frame
        end_time = time.time()
        frame_time = end_time - start_time
        if frame_time < 1.0 / frame_rate:
            time.sleep(1.0 / frame_rate - frame_time)  # Sleep to maintain frame rate

        # Exit on 'q' Key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show the entire path in a new window
    path_frame = np.zeros_like(frame, dtype=np.uint8)  # Correct initialization as NumPy array

    # Draw the entire path in green (from start to end)
    for (y, x) in all_path_points:
        pixel_x = x * grid_size + grid_size // 2
        pixel_y = y * grid_size + grid_size // 2
        cv2.circle(path_frame, (pixel_x, pixel_y), 1, (0, 255, 0), -1)

    # Display the detected objects in red
    for _, row in detected_objects.iterrows():
        cv2.rectangle(path_frame, (int(row['xmin']), int(row['ymin'])),
                      (int(row['xmax']), int(row['ymax'])), (0, 0, 255), 2)  # Red bounding box
        cv2.putText(path_frame, row["name"], (int(row['xmin']), int(row['ymin']) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

    # Show the entire path in a new window
    cv2.imshow("Full Path", path_frame)

    # Wait for 'q' key press to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()

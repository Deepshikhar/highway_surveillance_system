import cv2
import torch
import numpy as np
import time
from absl import app
from deep_sort_realtime.deepsort_tracker import DeepSort
import streamlit as st
import plotly.graph_objects as go
from ultralytics import YOLO
import tempfile
import os

### Configuration
st.set_page_config(
    page_title="Highway Surveillance",        
    layout="wide",
)

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)

def show_fps(frame, fps):    
    x, y, w, h = 10, 10, 330, 45

    # Draw black background rectangle
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), -1)

    # Text FPS
    cv2.putText(frame, "FPS: " + str(fps), (20,52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0,255,0), 3)

def show_counter(frame, title, class_names, vehicle_count, x_init):
    overlay = frame.copy()

    # Show Counters
    y_init = 100
    gap = 30

    alpha = 0.5

    cv2.rectangle(overlay, (x_init - 5, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for vehicle_id, count in vehicle_count.items():
        y_init += gap

        vehicle_name = class_names[vehicle_id]
        vehicle_count = "%.3i" % (count)
        cv2.putText(frame, vehicle_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)            
        cv2.putText(frame, vehicle_count, (x_init + 135, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def show_region(frame, points):
    for id, point in enumerate(points):        
        start_point = (int(points[id-1][0]), int(points[id-1][1]))
        end_point = (int(point[0]), int(point[1]))

        cv2.line(frame, start_point, end_point, (0,0,255), 3)  

def transform_points(perspective, points):
    if points.size == 0:
        return points

    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(
            reshaped_points, perspective)
    
    return transformed_points.reshape(-1, 2)     

def add_position_time(track_id, current_position, track_data):
    track_time = time.time()

    if(track_id in track_data):
        track_data[track_id]['position'].append(current_position)
    else:
        track_data[track_id] = {'position' : [current_position], 'time': track_time}

def calculate_speed(start, end, start_time):
    now = time.time()

    move_time = now - start_time    
    distance = abs(end - start)    
    distance = distance / 10

    # m/s
    speed = (distance / move_time)
    # Convert m/s to km/h
    speed = speed * 3.6

    return speed

def speed_estimation(vehicle_position, speed_region, perspective_region, track_data, track_id, text):   
    min_x = int(np.amin(speed_region[:, 0]))
    max_x = int(np.amax(speed_region[:, 0]))

    min_y = int(np.amin(speed_region[:, 1]))
    max_y = int(np.amax(speed_region[:, 1]))

    speed = 0

    if((vehicle_position[0] in range(min_x, max_x)) and (vehicle_position[1] in range(min_y, max_y))):
        points = np.array([[vehicle_position[0], vehicle_position[1]]], 
                        dtype=np.float32)                                

        point_transform = transform_points(perspective_region, points)                
        
        add_position_time(track_id, int(point_transform[0][1]), track_data)                

        if(len(track_data[track_id]['position']) > 5):
            start_position = track_data[track_id]['position'][0]
            end_position = track_data[track_id]['position'][-1]
            start_estimate = track_data[track_id]['time']

            speed = calculate_speed(start_position, end_position, start_estimate)
            speed_string = "{:.2f}".format(speed)

            text = text + " - " + speed_string + " km/h"
    
    return text, speed

def process_video(video_source, tracker, model, class_names, colors, result_elem):
    # Initialize the video capture
    if isinstance(video_source, str) and os.path.exists(video_source):
        # Video file path
        cap = cv2.VideoCapture(video_source)
    elif hasattr(video_source, 'name'):
        # Uploaded file object
        cap = cv2.VideoCapture(video_source.name)
    else:
        st.error('Error: Invalid video source.')
        return

    if not cap.isOpened():
        st.error('Error: Unable to open video source.')
        return   

    ## Vehicle Counter
    # Helper Variable
    entered_vehicle_ids = []
    exited_vehicle_ids = []    

    vehicle_class_ids = [1, 2, 3, 5, 7]  # COCO class IDs for vehicles

    vehicle_entry_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    vehicle_exit_count = {
        1: 0,  # bicycle
        2: 0,  # car
        3: 0,  # motorcycle
        5: 0,  # bus
        7: 0   # truck
    }
    
    entry_line = {
        'x1' : 160, 
        'y1' : 558,  
        'x2' : 708,  
        'y2' : 558,          
    }
    exit_line = {
        'x1' : 1155, 
        'y1' : 558,  
        'x2' : 1718,  
        'y2' : 558,          
    }
    offset = 20
    ##

    ## Speed Estimation
    # Region 1 (Left)
    speed_region_1 = np.float32([[393, 478], 
                    [760, 482],
                    [611, 838], 
                    [-135, 777]]) 
    width = 150
    height = 270
    target_1 = np.float32([[0, 0], 
                    [width, 0],
                    [width, height], 
                    [0, height]])
    
    # Region 2 (Right)
    speed_region_2 = np.float32([[1074, 500], 
                    [1422, 490],
                    [2021, 812], 
                    [1377, 932]])     
    width = 120
    height = 270
    target_2 = np.float32([[0, 0], 
                    [width, 0],
                    [width, height], 
                    [0, height]])
    
    # Transform Perspective
    perspective_region_1 = cv2.getPerspectiveTransform(speed_region_1, target_1)    
    perspective_region_2 = cv2.getPerspectiveTransform(speed_region_2, target_2)

    track_data = {}

    # Create placeholders for the chart and statistics
    chart_placeholder = st.empty()
    stats_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Run YOLOv11 model on each frame
        results = model(frame, verbose=False)  # verbose=False to reduce output
        
        # Counting Line
        cv2.line(frame, (entry_line['x1'], entry_line['y1']), (exit_line['x2'], exit_line['y2']), (0, 127, 255), 3)

        # Speed Region
        show_region(frame, speed_region_1)
        show_region(frame, speed_region_2)

        detect = []
        # Process YOLOv11 results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Filter out weak detections by confidence threshold
                    if confidence < 0.5:
                        continue

                    detect.append([[int(x1), int(y1), int(x2-x1), int(y2-y1)], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        # Average Speed
        speeds = []

        for track in tracks:
            if not track.is_confirmed():
                continue            
            
            track_id = track.track_id
            ltrb = track.to_ltrb()            
            x1, y1, x2, y2 = map(int, ltrb)    
            class_id = track.get_det_class()
            color = colors[class_id]
            B, G, R = map(int, color)

            text = f"{track_id} - {class_names[class_id]}"            
            
            center_x = int((x1 + x2) / 2 )
            center_y = int((y1 + y2) / 2 )

            ## Speed Estimation
            # Region 1  
            vehicle_position = (center_x, y2)
            text, vehicle_speed = speed_estimation(vehicle_position, speed_region_1, perspective_region_1, track_data, track_id, text)   

            if(vehicle_speed > 0):
                speeds.append(vehicle_speed)

            # Region 2  
            vehicle_position = (center_x, y1)
            text, vehicle_speed = speed_estimation(vehicle_position, speed_region_2, perspective_region_2, track_data, track_id, text)            

            if(vehicle_speed > 0):
                speeds.append(vehicle_speed)
            ##

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Counter in
            if((center_x in range(entry_line['x1'], entry_line['x2'])) and (center_y in range(entry_line['y1'], entry_line['y1'] + offset)) ):            
                if(int(track_id) not in entered_vehicle_ids and class_id in vehicle_class_ids):                    
                    vehicle_entry_count[class_id] += 1                    
                    entered_vehicle_ids.append(int(track_id))                

            # Counter out
            if((center_x in range(exit_line['x1'], exit_line['x2'])) and (center_y in range(exit_line['y1'] - offset, exit_line['y1'])) ):                        
                if(int(track_id) not in exited_vehicle_ids and class_id in vehicle_class_ids):                    
                    vehicle_exit_count[class_id] += 1                                      
                    exited_vehicle_ids.append(int(track_id)) 
        
        # Show Counters
        show_counter(frame, "Vehicle Enter", class_names, vehicle_entry_count, 10)
        show_counter(frame, "Vehicle Exit", class_names, vehicle_exit_count, 1710)

        end_time = time.time()

        # FPS Calculation
        fps = 1 / (end_time - start_time)
        fps = float("{:.2f}".format(fps))                

        resized = cv2.resize(frame, (1280, 720))

        # Average Speed        
        total_speed = sum(speeds)
        num_speeds = len(speeds)
        average_speed = 0
        if(num_speeds > 0):
            average_speed = total_speed / num_speeds

        average_speed = "{:.2f} km/h".format(average_speed)  

        # Total Entered and Exited Vehicles
        all_vehicle_entry_count = sum(vehicle_entry_count.values())
        all_vehicle_exit_count = sum(vehicle_exit_count.values())
        
        # Combine the counts
        vehicle_count = {}
        for key in vehicle_entry_count:
            vehicle_count[key] = vehicle_entry_count[key] + vehicle_exit_count[key]

        with result_elem.container():    
            # Create main layout with better proportions
            main_col1, main_col2 = st.columns((0.45, 0.55))

            with main_col1:
                # Video section
                st.markdown("### Vehicle Detection")
                st.image(resized, channels="BGR", use_container_width=True)    
                
                # Metrics row
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric(label="FPS", value=fps)
                with metrics_col2:
                    st.metric(label="Vehicle Enter", value=all_vehicle_entry_count)
                with metrics_col3:
                    st.metric(label="Vehicle Exit", value=all_vehicle_exit_count)
                with metrics_col4:
                    st.metric(label="Average Speed", value=average_speed)

            with main_col2:
                # Statistics section - more compact
                st.markdown("### Vehicle Statistics")
                
                # Bar chart with compact layout
                with st.container():
                    vibrant_colors = ['rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 205, 86)',
                                    'rgb(75, 192, 192)', 'rgb(153, 102, 255)']
                    
                    labels = ["bicycle", "car", "motorbike", "bus", "truck"]
                    total_vehicle_counts = list(vehicle_count.values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=labels, 
                        y=total_vehicle_counts,
                        marker_color=vibrant_colors,
                        text=total_vehicle_counts,
                        textposition='auto',
                        textfont=dict(size=10)
                    )])
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=30, b=20),
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=9, color="white"),
                        xaxis=dict(tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{st.session_state.frame_count}")
                
                # Detailed counts in a more compact way
                st.markdown("#### Detailed Counts")
                detail_cols = st.columns(5)
                all_vehicle_count = all_vehicle_entry_count + all_vehicle_exit_count
                
                for index, (key, value) in enumerate(vehicle_count.items()):
                    with detail_cols[index]:
                        number_of_vehicle = '{:,.0f}'.format(value)
                        vehicle_percentage = (value / all_vehicle_count * 100) if all_vehicle_count > 0 else 0
                        formatted_percentage = "{:.1f}%".format(vehicle_percentage)
                        
                        st.metric(
                            label=labels[index].title(),
                            value=number_of_vehicle,
                            delta=formatted_percentage,
                            delta_color="off"
                        )

        # Increment frame count for unique keys
        st.session_state.frame_count += 1

    # Release video capture
    cap.release()

def main(_argv):
    ## Dashboard    
    st.title("Highway Vehicle Surveillance with YOLOv11")    
    
    # Initialize session state for tracking
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    result_elem = st.empty()
    
    # Video input options
    st.sidebar.title("Video Input Options")
    
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["File Path", "Upload Video File"]
    )
    
    video_source = None
    
    if input_method == "File Path":
        video_path = st.sidebar.text_input(
            "Enter video file path:",
            value="highway.mp4",
            help="Enter the path to your video file (e.g., 'video.mp4', 'path/to/video.avi')"
        )
        
        if st.sidebar.button("Process Video") and video_path:
            if os.path.exists(video_path):
                video_source = video_path
                st.session_state.processing = True
            else:
                st.sidebar.error(f"File not found: {video_path}")
    
    else:  # Upload Video File
        uploaded_file = st.sidebar.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Upload a video file for processing"
        )
        
        if uploaded_file is not None and st.sidebar.button("Process Uploaded Video"):
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_source = tmp_file.name
                st.session_state.processing = True
    
    # Stop processing button
    if st.session_state.processing:
        if st.sidebar.button("Stop Processing"):
            st.session_state.processing = False
            st.rerun()
    
    # Initialize models only once
    if 'tracker' not in st.session_state:
        st.session_state.tracker = DeepSort(max_age=5)
    
    if 'model' not in st.session_state:
        yolov11_weights = "yolo11s.pt"
        st.session_state.model = YOLO(yolov11_weights)
        st.session_state.class_names = st.session_state.model.names
        np.random.seed(42)
        st.session_state.colors = np.random.randint(0, 255, size=(len(st.session_state.class_names), 3))
    
    # Process video if source is available and processing is enabled
    if st.session_state.processing and video_source is not None:
        with st.spinner("Processing video..."):
            process_video(
                video_source, 
                st.session_state.tracker, 
                st.session_state.model, 
                st.session_state.class_names, 
                st.session_state.colors, 
                result_elem
            )
        
        # Clean up temporary file if it was an upload
        if input_method == "Upload Video File" and os.path.exists(video_source):
            os.unlink(video_source)
        
        st.session_state.processing = False
        st.success("Video processing completed!")
    
    elif not st.session_state.processing:
        # Show instructions when not processing
        with result_elem.container():
            st.info("👆 Please select a video input method from the sidebar and click 'Process Video' to start analysis.")

if __name__ == '__main__':
    app.run(main)

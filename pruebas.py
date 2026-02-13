from ultralytics import YOLO
import cv2
import time
import torch

def process_video_with_yolo(model_path, video_path):
    """
    Processes a video with a YOLOv8 segmentation model for real-time inference.

    Args:
        model_path (str): Path to the YOLOv8 model weights (e.g., 'best.pt').
        video_path (str): Path to the input video file.
    """ 
    model = YOLO(model_path)

    if torch.cuda.is_available():
        model.model.to('cuda').half() 
        device = 'cuda'
        print("Using GPU for inference.")
    else:
        device = 'cpu'
        print("Using CPU for inference.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: FPS is 0, defaulting to 30 FPS.")
        fps = 30 
    frame_display_time_ms = int((1 / fps) * 1000) 

    print(f"Processing video with {fps:.2f} FPS.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        input_width, input_height = 320, 192 
        resized_frame = cv2.resize(frame, (input_width, input_height))

        start_time = time.time()
        with torch.no_grad():
            results = model(resized_frame, device=device, verbose=False) 
        inference_time = time.time() - start_time

        annotated_frame = results[0].plot()

        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

        cv2.putText(annotated_frame, f"Inferencia: {inference_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Detección de Plecostomus", annotated_frame)
        actual_delay_ms = max(1, frame_display_time_ms - int(inference_time * 1000))
        
        if cv2.waitKey(actual_delay_ms) & 0xFF == ord('q'):
            print("Exiting video playback.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == '__main__':
    # Modelo entrenado 
    model_weights_path = r'C:\Users\diego\Desktop\plecostomus_project\plecostomus-seg\yolov8n-seg-custom3\weights\best.pt'
    #video_file_path = r'C:\Users\diego\Desktop\plecostomus_project\videos\Video de WhatsApp 2025-06-09 a las 16.18.38_0ef609fe.mp4'
    #video prueba
    video_file_path = r'C:\Users\diego\Desktop\plecostomus_project\videos\Video de WhatsApp 2025-06-09 a las 16.34.27_371a8f44.mp4'
    #video_file_path = r'C:\Users\diego\Desktop\plecostomus_project\videos\Video de WhatsApp 2025-06-09 a las 16.57.02_2e4b6cb5.mp4'
    process_video_with_yolo(model_weights_path, video_file_path)
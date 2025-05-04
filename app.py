from flask import Flask, render_template, request, Response, stream_with_context, redirect, url_for, jsonify
import cv2
import threading
import queue  # Import queue for speech handling
import pyttsx3  # Import the text-to-speech library
import time  # Import time for sleep
import atexit  # Import atexit for cleanup
import speech_recognition as sr  # Import speech recognition 
import requests  # Import requests for internal API calls
import json  # Import json for formatting event data
from components import load_depth_model, detect_objects, estimate_depth
from ultralytics import YOLO  # Import YOLO from ultralytics

app = Flask(__name__)

stop_loop = False  # Shared variable to control the loop
speech_queue = queue.Queue()  # Queue for managing speech requests
voice_command_active = False  # Flag to control voice command thread

# Global variables for models
object_detection_model = None
depth_estimation_model = None

# Speech loop to handle text-to-speech sequentially
def speech_loop():
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()  # Wait until a message is available
        if text == "__EXIT__":
            break  # Exit the loop if the special exit message is received
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Voice command recognition thread - FIXED IMPLEMENTATION
def voice_command_loop():
    global stop_loop, voice_command_active
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise once at startup
    with sr.Microphone() as source:
        speech_queue.put("Calibrating microphone for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        speech_queue.put("Voice commands activated")
        
        # Notify frontend
        voice_event_queue.put(json.dumps({
            "action": "voice_activated",
            "message": "Voice command system is listening"
        }))
    
    while voice_command_active:
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
            
            try:
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio).lower()
                
                # Log the recognized command
                voice_event_queue.put(json.dumps({
                    "action": "voice_recognized",
                    "message": f"Recognized: {text}"
                }))
                
                # Process commands
                if "start" in text or "begin" in text or "detect" in text:
                    speech_queue.put("Starting object detection")
                    requests.get('http://localhost:5000/voice_start')
                
                elif "stop" in text or "end" in text or "halt" in text:
                    speech_queue.put("Stopping object detection")
                    requests.post('http://localhost:5000/stop')
                
                elif "exit" in text or "quit" in text:
                    speech_queue.put("Deactivating voice commands")
                    voice_command_active = False
                
            except sr.UnknownValueError:
                # Speech was unintelligible
                pass
            except sr.RequestError as e:
                speech_queue.put(f"Could not request results; {e}")
                
        except sr.WaitTimeoutError:
            # Timeout occurred, just continue the loop
            pass
        except Exception as e:
            speech_queue.put(f"Voice command error: {str(e)}")
            print(f"Voice command error: {str(e)}")
    
    # Notify that voice commands are deactivated
    voice_event_queue.put(json.dumps({
        "action": "voice_deactivated",
        "message": "Voice command system stopped"
    }))
    speech_queue.put("Voice command system deactivated")

# Start the speech thread
speech_thread = threading.Thread(target=speech_loop, daemon=True)
speech_thread.start()

# Voice command thread will be started when requested

@app.route('/')
def index():
    global object_detection_model, depth_estimation_model
    
    # Load models if not already loaded
    if depth_estimation_model is None:
        depth_estimation_model = load_depth_model()  # Load the MiDaS depth estimation model
    
    if object_detection_model is None:
        object_detection_model = YOLO('yolov8n.pt')  # Load the YOLOv8n model
        
    return render_template('index.html')

# Queue for voice command events
voice_event_queue = queue.Queue()

@app.route('/voice_events')
def voice_events():
    """Server-sent events endpoint for voice command notifications"""
    def generate():
        yield "data: {\"message\": \"Connected to voice command events\"}\n\n"
        
        while True:
            try:
                # Get event from queue with timeout
                event = voice_event_queue.get(timeout=1)
                yield f"data: {event}\n\n"
            except queue.Empty:
                # Send a keep-alive comment to prevent timeout
                yield ": keep-alive\n\n"
            except GeneratorExit:
                # Client disconnected
                break
    
    return Response(stream_with_context(generate()), content_type="text/event-stream")

@app.route('/voice_command', methods=['POST'])
def voice_command():
    global voice_command_active
    
    action = request.json.get('action', '')
    
    if action == 'start':
        # Start voice command recognition if not already running
        if not voice_command_active:
            voice_command_active = True
            voice_thread = threading.Thread(target=voice_command_loop, daemon=True)
            voice_thread.start()
            return jsonify({"status": "Voice command mode activated"})
        else:
            return jsonify({"status": "Voice command already active"})
    
    elif action == 'stop':
        # Stop voice command recognition
        voice_command_active = False
        return jsonify({"status": "Voice command mode deactivated"})
    
    return jsonify({"status": "Invalid action"})

@app.route('/voice_start')
def voice_start():
    """Endpoint for voice commands to trigger start without browser UI interaction"""
    # This will be called by the voice command thread
    global stop_loop
    stop_loop = False
    
    # Add event to queue to notify frontend
    voice_event_queue.put(json.dumps({
        "action": "start_detection",
        "message": "Voice command: Starting detection"
    }))
    
    # Use the same logic as the regular start endpoint
    return start()

@app.route('/start')
def start():
    global stop_loop
    stop_loop = False  # Reset the stop flag
    
    def process_loop():
        global stop_loop
        
        # Send initial message
        yield "data: Detection process started\n\n"
        
        while not stop_loop:
            try:
                cap = cv2.VideoCapture(0)  # Open the default camera
                ret, frame = cap.read()    # Read a frame
                cap.release()              # Release the camera
                
                if not ret:
                    message = "Failed to capture image!"
                    speech_queue.put(message)  # Add message to the speech queue
                    yield f"data: {message}\n\n"
                    continue
                
                object_list = detect_objects(frame, object_detection_model)
                
                if object_list:
                    object_depths = estimate_depth(frame, object_list, depth_estimation_model)
                    depth = float(object_depths['depth'])  # Ensure depth is a float
                    formatted_depth = f"{depth:.2f}"  # Format depth to 2 decimal places
                    message = f"{object_depths['class_name']} detected at {formatted_depth} centimeters"
                    speech_queue.put(message)  # Add message to the speech queue
                    yield f"data: {message}\n\n"
                else:
                    message = "No objects detected"
                    speech_queue.put(message)  # Add message to the speech queue
                    yield f"data: {message}\n\n"
                
                time.sleep(3)  # Pause for 3 seconds before the next iteration
            
            except Exception as e:
                error_message = f"Error: {str(e)}"
                yield f"data: {error_message}\n\n"
                speech_queue.put("System error occurred")
        
        yield "data: Detection process stopped\n\n"
    
    return Response(stream_with_context(process_loop()), 
                   content_type='text/event-stream')

@app.route('/stop', methods=['POST'])
def stop():
    global stop_loop
    stop_loop = True  # Set the stop flag to break the loop
    return jsonify({"status": "stopped"})

# Cleanup on exit
@atexit.register
def shutdown():
    global voice_command_active
    
    # Stop voice command thread
    voice_command_active = False
    
    # Close speech engine
    speech_queue.put("__EXIT__")  # Signal the speech thread to exit
    speech_thread.join(timeout=1)  # Wait for the speech thread to finish

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
#!/usr/bin/env python3
"""
Emergency Detection System
Detects explosions, fires, and crowd panic using computer vision
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class EmergencyDetector:
    def __init__(self):
        # Camera setup
        self.cap = None
        self.running = False
        
        # Detection parameters
        self.motion_threshold = 20000  # Threshold for significant motion
        self.fire_orange_lower = np.array([10, 150, 150], dtype=np.uint8)  # More strict
        self.fire_orange_upper = np.array([20, 255, 255], dtype=np.uint8)
        self.fire_yellow_lower = np.array([20, 150, 150], dtype=np.uint8)  # More strict
        self.fire_yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
        
        # Motion history
        self.motion_history = deque(maxlen=30)  # Last 30 frames
        self.prev_frame = None
        
        # Alert state
        self.alert_active = False
        self.alert_type = None
        self.alert_start_time = None
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
        # Optical flow parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def initialize_camera(self, source=0):
        """Initialize camera/video source"""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print("Error: Could not open camera/video")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def detect_fire_explosion(self, frame):
        """Detect fire/explosion based on color analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for fire colors (orange and yellow)
        mask_orange = cv2.inRange(hsv, self.fire_orange_lower, self.fire_orange_upper)
        mask_yellow = cv2.inRange(hsv, self.fire_yellow_lower, self.fire_yellow_upper)
        
        # Combine masks
        fire_mask = cv2.bitwise_or(mask_orange, mask_yellow)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate fire area
        fire_area = cv2.countNonZero(fire_mask)
        total_area = frame.shape[0] * frame.shape[1]
        fire_percentage = (fire_area / total_area) * 100
        
        return fire_percentage > 5, fire_percentage, fire_mask  # Alert if >30% of frame
    
    def detect_motion_intensity(self, frame):
        """Detect intense motion that could indicate panic or explosion"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0, None
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate motion amount
        motion_amount = cv2.countNonZero(thresh)
        
        # Store motion history
        self.motion_history.append(motion_amount)
        
        # Check for sudden spike in motion (possible explosion/panic)
        is_alert = False
        if len(self.motion_history) > 30:
            recent_avg = np.mean(list(self.motion_history)[-10:])
            if recent_avg > self.motion_threshold:
                is_alert = True
        
        self.prev_frame = gray
        return is_alert, motion_amount, thresh
    
    def detect_crowd_panic(self, frame):
        """Detect crowd panic using optical flow (people running)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0, None
        
        # Detect features to track
        p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **self.feature_params)
        
        if p0 is None:
            self.prev_frame = gray
            return False, 0, None
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, p0, None, **self.lk_params)
        
        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            # Calculate motion vectors
            motion_vectors = good_new - good_old
            speeds = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
            
            # Check for fast movement (running)
            fast_movement_count = np.sum(speeds > 5)  # Pixels per frame
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
            
            # Alert if many points moving fast
            is_panic = fast_movement_count > 20 and avg_speed > 3
            
            return is_panic, avg_speed, motion_vectors
        
        return False, 0, None
    
    def trigger_alarm(self, alert_type):
        """Trigger visual and audio alarm"""
        if not self.alert_active:
            self.alert_active = True
            self.alert_type = alert_type
            self.alert_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"🚨 ALERT: {alert_type.upper()} DETECTED! 🚨")
            print(f"{'='*50}\n")
            
            # Start alarm sound in separate thread
            threading.Thread(target=self.play_alarm_sound, daemon=True).start()
    
    def play_alarm_sound(self):
        """Play alarm sound (visual indicator since audio might not work)"""
        for _ in range(5):
            print("\a")  # System beep
            time.sleep(0.5)
    
    def reset_alarm(self):
        """Reset alarm state"""
        if self.alert_active and time.time() - self.alert_start_time > 5:
            self.alert_active = False
            self.alert_type = None
            print("\n[System returned to normal monitoring]\n")
    
    def process_frame(self, frame):
        """Process a single frame for all detection methods"""
        detection_frame = frame.copy()
        
        # Run detections
        fire_detected, fire_pct, fire_mask = self.detect_fire_explosion(frame)
        motion_alert, motion_amount, motion_mask = self.detect_motion_intensity(frame)
        panic_detected, avg_speed, motion_vectors = self.detect_crowd_panic(frame)
        
        # Only trigger alert if conditions are met
        if fire_detected:
            self.trigger_alarm("FIRE/EXPLOSION")
        
        if motion_alert:
            self.trigger_alarm("INTENSE MOTION")
        
        if panic_detected:
            self.trigger_alarm("CROWD PANIC")
        
        # Show alert ONLY when conditions are met
        if self.alert_active:
            # Red border
            cv2.rectangle(detection_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            # Emergency message
            cv2.putText(detection_frame, "!!! EMERGENCY DETECTED !!!", 
                       (frame.shape[1]//2 - 250, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Show what was detected (smaller text at top)
            y_offset = 30
            if fire_detected:
                cv2.putText(detection_frame, f"FIRE/EXPLOSION: {fire_pct:.1f}%", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            
            if motion_alert:
                cv2.putText(detection_frame, f"HIGH MOTION: {motion_amount}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            
            if panic_detected:
                cv2.putText(detection_frame, f"CROWD PANIC: Speed {avg_speed:.1f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Reset alarm after calm period
            self.reset_alarm()
            cv2.rectangle(detection_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
        
        return detection_frame
    
    def run(self, source=0):
        """Main detection loop"""
        if not self.initialize_camera(source):
            return
        
        self.running = True
        print("Emergency Detection System Started")
        print("Press 'q' to quit")
        print("Press 's' to take screenshot")
        print("-" * 50)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or camera error")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Display FPS
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (processed_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Emergency Detection System', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("System stopped")

class EmergencyDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emergency Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a1a")
        
        self.detector = EmergencyDetector()
        self.running = False
        self.video_source = None
        
        # Header
        header = tk.Frame(root, bg="#0d47a1", height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🚨 Emergency Detection System", 
                        font=("Arial", 20, "bold"), 
                        bg="#0d47a1", fg="white")
        title.pack(pady=15)
        
        # Control panel
        control_frame = tk.Frame(root, bg="#2b2b2b", pady=10)
        control_frame.pack(fill=tk.X)
        
        # Buttons
        btn_style = {"font": ("Arial", 11), "width": 15, "height": 2}
        
        self.btn_webcam = tk.Button(control_frame, text="📷 Start Webcam", 
                                    command=self.start_webcam, 
                                    bg="#4CAF50", fg="white", **btn_style)
        self.btn_webcam.pack(side=tk.LEFT, padx=10)
        
        self.btn_video = tk.Button(control_frame, text="🎥 Load Video", 
                                   command=self.load_video, 
                                   bg="#2196F3", fg="white", **btn_style)
        self.btn_video.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(control_frame, text="⏹ Stop", 
                                  command=self.stop, 
                                  bg="#f44336", fg="white", 
                                  state=tk.DISABLED, **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        self.btn_screenshot = tk.Button(control_frame, text="📸 Screenshot", 
                                        command=self.take_screenshot, 
                                        bg="#FF9800", fg="white", 
                                        state=tk.DISABLED, **btn_style)
        self.btn_screenshot.pack(side=tk.LEFT, padx=10)
        
        # Video display area
        video_container = tk.Frame(root, bg="#000000", relief=tk.SUNKEN, bd=3)
        video_container.pack(expand=True, fill=tk.BOTH, padx=15, pady=10)
        
        self.video_label = tk.Label(video_container, bg="black", 
                                     text="No video source\nClick 'Start Webcam' or 'Load Video'",
                                     font=("Arial", 14), fg="gray")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Alert panel (starts hidden)
        self.alert_frame = tk.Frame(root, bg="#4CAF50", height=60)
        self.alert_frame.pack(fill=tk.X)
        self.alert_frame.pack_propagate(False)
        
        self.alert_label = tk.Label(self.alert_frame, 
                                     text="✓ System Normal - Monitoring", 
                                     font=("Arial", 16, "bold"), 
                                     bg="#4CAF50", fg="white")
        self.alert_label.pack(expand=True)
        
        # Status bar
        status_frame = tk.Frame(root, bg="#424242")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready • No source active")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               bg="#424242", fg="white", 
                               font=("Arial", 10), anchor=tk.W, padx=10)
        status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = tk.Label(status_frame, textvariable=self.fps_var, 
                            bg="#424242", fg="white", 
                            font=("Arial", 10), anchor=tk.E, padx=10)
        fps_label.pack(side=tk.RIGHT)
        
        # FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def start_webcam(self):
        if self.detector.initialize_camera(0):
            self.running = True
            self.video_source = "Webcam"
            self.status_var.set("Active • Webcam")
            self.btn_webcam.config(state=tk.DISABLED)
            self.btn_video.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_screenshot.config(state=tk.NORMAL)
            self.update_frame()
        else:
            messagebox.showerror("Error", "Could not open webcam")
    
    def load_video(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            if self.detector.initialize_camera(filepath):
                self.running = True
                self.video_source = filepath.split('/')[-1]
                self.status_var.set(f"Active • {self.video_source}")
                self.btn_webcam.config(state=tk.DISABLED)
                self.btn_video.config(state=tk.DISABLED)
                self.btn_stop.config(state=tk.NORMAL)
                self.btn_screenshot.config(state=tk.NORMAL)
                self.update_frame()
            else:
                messagebox.showerror("Error", f"Could not open video:\n{filepath}")
    
    def stop(self):
        self.running = False
        if self.detector.cap:
            self.detector.cap.release()
        self.detector.prev_frame = None
        self.detector.motion_history.clear()
        
        self.status_var.set("Ready • No source active")
        self.video_label.config(image='', 
                               text="No video source\nClick 'Start Webcam' or 'Load Video'")
        self.btn_webcam.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_screenshot.config(state=tk.DISABLED)
        self.reset_alert()
    
    def take_screenshot(self):
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Screenshot Saved", f"Saved as:\n{filename}")
    
    def update_frame(self):
        if not self.running:
            return
        
        ret, frame = self.detector.cap.read()
        if not ret:
            self.stop()
            messagebox.showinfo("Video Ended", "Video playback completed")
            return
        
        # Resize to fixed resolution
        frame = cv2.resize(frame, (640, 480))
        self.current_frame = frame.copy()
        
        # Process frame with detector
        processed = self.detector.process_frame(frame)
        
        # Update alert status
        if self.detector.alert_active:
            self.alert_frame.config(bg="#f44336")
            alert_text = f"⚠️ ALERT: {self.detector.alert_type} DETECTED ⚠️"
            self.alert_label.config(bg="#f44336", text=alert_text)
        else:
            self.alert_frame.config(bg="#4CAF50")
            self.alert_label.config(bg="#4CAF50", 
                                    text="✓ System Normal - Monitoring")
        
        # Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30 / elapsed if elapsed > 0 else 0
            self.fps_var.set(f"FPS: {self.current_fps:.1f}")
            self.fps_start_time = time.time()
        
        # Convert for Tkinter display
        cv_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text='')
        
        # Schedule next frame (30 FPS = ~33ms delay)
        self.root.after(33, self.update_frame)
    
    def reset_alert(self):
        self.alert_frame.config(bg="#4CAF50")
        self.alert_label.config(bg="#4CAF50", 
                                text="✓ System Normal - Monitoring")
        self.detector.alert_active = False
        self.detector.alert_type = None
    
    def on_closing(self):
        if self.running:
            if messagebox.askokcancel("Quit", "Detection is active. Are you sure you want to quit?"):
                self.stop()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main function with GUI"""
    root = tk.Tk()
    app = EmergencyDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

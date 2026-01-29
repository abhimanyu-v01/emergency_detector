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

def main():
    """Main function"""
    print("="*60)
    print("Emergency Detection System for Public Safety")
    print("="*60)
    print("\nThis system detects:")
    print("  • Fire and explosions (color-based)")
    print("  • Intense motion (possible panic or blast)")
    print("  • Crowd panic (people running)")
    print("\nOptions:")
    print("  1. Use webcam (default)")
    print("  2. Use video file")
    print("="*60)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    detector = EmergencyDetector()
    
    if choice == "2":
        video_path = input("Enter video file path: ").strip()
        # Remove quotes if user included them
        video_path = video_path.strip("'\"")
        print(f"Attempting to open: {video_path}")
        detector.run(video_path)
    else:
        detector.run(0)  # Default webcam

if __name__ == "__main__":
    main()

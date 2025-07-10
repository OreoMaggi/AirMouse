import cv2
import mediapipe as mp
import pyautogui
import time
import math
import threading
import tkinter as tk
from tkinter import StringVar

# === Gesture Recognition Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# === Gesture State Variables ===
dragging = False
left_pinching = False
right_pinching = False
gesture_control_enabled = True
click_start_time = 0
CLICK_HOLD_THRESHOLD = 0.3  # shorter delay for faster drag start

def calc_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# === Toggle Control from GUI ===
def toggle_gesture_control():
    global gesture_control_enabled
    gesture_control_enabled = not gesture_control_enabled
    toggle_btn.config(text="Disable Control" if gesture_control_enabled else "Enable Control")
    if 'gesture_text' in globals():
        gesture_text.set("Gesture: Control ON" if gesture_control_enabled else "Gesture: Control OFF")

# === Gesture Recognition Main Loop ===
def run_gesture_control():
    global dragging, left_pinching, right_pinching, click_start_time

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_gesture = "Idle"

        if result.multi_hand_landmarks and gesture_control_enabled:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index = (int(lm[8].x * w), int(lm[8].y * h))
                thumb = (int(lm[4].x * w), int(lm[4].y * h))
                middle = (int(lm[12].x * w), int(lm[12].y * h))
                screen_index = (int(lm[8].x * screen_width), int(lm[8].y * screen_height))

                dist_index_thumb = calc_distance(index, thumb)
                dist_middle_thumb = calc_distance(middle, thumb)

                # Move Mouse (if not dragging)
                if dist_index_thumb >= 40 and not dragging:
                    pyautogui.moveTo(screen_index[0], screen_index[1])

                # --- Left Click or Drag ---
                if dist_index_thumb < 40:
                    if not left_pinching:
                        left_pinching = True
                        click_start_time = time.time()
                    else:
                        if not dragging:
                            if time.time() - click_start_time > CLICK_HOLD_THRESHOLD:
                                pyautogui.mouseDown()
                                dragging = True
                                current_gesture = "Dragging"
                        else:
                            # Continue dragging smoothly
                            pyautogui.moveTo(screen_index[0], screen_index[1])
                            current_gesture = "Dragging"
                else:
                    if left_pinching:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False
                        elif time.time() - click_start_time < CLICK_HOLD_THRESHOLD:
                            pyautogui.click()
                            current_gesture = "Left Click"
                        left_pinching = False

                # --- Right Click ---
                if dist_middle_thumb < 40:
                    if not right_pinching:
                        right_pinching = True
                        pyautogui.click(button='right')
                        current_gesture = "Right Click"
                else:
                    right_pinching = False

                # Draw Landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update gesture text if GUI has launched
        if 'gesture_text' in globals():
            gesture_text.set(f"Gesture: {current_gesture}")

        # Overlay gesture on webcam feed
        cv2.putText(frame, current_gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        cv2.imshow("Gesture Mouse", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# === GUI ===
def start_gui():
    global toggle_btn, gesture_text

    window = tk.Tk()
    window.title("Gesture Mouse GUI")
    window.geometry("300x140")

    gesture_text = StringVar()
    gesture_text.set("Gesture: Idle")

    lbl = tk.Label(window, textvariable=gesture_text, font=("Helvetica", 14))
    lbl.pack(pady=10)

    toggle_btn = tk.Button(window, text="Disable Control", font=("Helvetica", 12), command=toggle_gesture_control)
    toggle_btn.pack(pady=10)

    tk.Label(window, text="Press ESC in camera window to exit", font=("Helvetica", 9)).pack()
    window.mainloop()

# === Start GUI + Gesture in Threads ===
if __name__ == "__main__":
    threading.Thread(target=start_gui, daemon=True).start()
    run_gesture_control()

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import serial
import random

# ========================
# CARGAR MODELO YOLO
# ========================
modelp = YOLO("yolov8n.pt")  # Se descarga automático
modelh = YOLO("best2.pt")  # Modelo personalizado para cascos

# ========================
# CONFIGURACIÓN SERIAL
# ========================
try:
    arduino = serial.Serial('COM3', 9600)
except:
    arduino = None
    print("Arduino no conectado")

# ========================
# VARIABLES
# ========================
max_personas = 0
cap = None
running = False

# ========================
# DETECCIÓN + TRACKING
# ========================
def detectar_personas(frame):
    global max_personas

    resultadosp = modelp.track(frame, persist=True, verbose=False)
    resultadosh = modelh.track(frame, persist=True, verbose=False)

    ids_activos = set()
    helmets_detected = set()

    for r in resultadosp:
        boxes = r.boxes

        if boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id):
                clase = int(box.cls[0])

                if clase == 0:  # Persona
                    track_id = int(track_id)
                    ids_activos.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (
                        (track_id * 50) % 255,
                        (track_id * 80) % 255,
                        (track_id * 110) % 255
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    
    for rh in resultadosh:
        boxesh = rh.boxes

        if boxesh.id is not None:
            for boxh, track_idh in zip(boxesh, boxesh.id):
                claseh = int(boxh.cls[0])

                if claseh == 0:  # Casco
                    track_idh = int(track_idh)
                    helmets_detected.add(track_idh)

                    x1, y1, x2, y2 = map(int, boxh.xyxy[0])
                    color = (
                        (track_idh * 50) % 255,
                        (track_idh * 80) % 255,
                        (track_idh * 110) % 255
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {track_idh} - Casco",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)

    total_personas = len(ids_activos)
    total_cascos = len(helmets_detected)

    cv2.putText(frame, f"Total Personas: {total_personas}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Total Cascos: {total_cascos}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Enviar señal si supera límite
    if total_personas > max_personas:
        if arduino:
            arduino.write(b"A")
        else:
            print("Límite excedido - Señal A enviada (simulada)")
            print(f"Personas: {total_personas} | Cascos: {total_cascos}")
    elif total_personas <= max_personas:
        if total_cascos < total_personas:
            print("Alerta: Personas sin casco detectadas")
            print(f"Personas: {total_personas} | Cascos: {total_cascos}")
        else:
            if arduino:
                arduino.write(b"B")
            else:
                print("Dentro del límite - Señal B enviada (simulada)")
                print(f"Personas: {total_personas} | Cascos: {total_cascos}")

    return frame

# ========================
# LOOP CÁMARA
# ========================
def update_frame():
    global cap, running

    if running:
        ret, frame = cap.read()
        if ret:
            frame = detectar_personas(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        video_label.after(10, update_frame)

# ========================
# INICIAR
# ========================
def iniciar():
    global max_personas, cap, running

    try:
        max_personas = int(entry.get())
    except:
        messagebox.showerror("Error", "Ingrese un número válido")
        return

    cap = cv2.VideoCapture(0)
    running = True
    update_frame()

# ========================
# INTERFAZ
# ========================
root = tk.Tk()
root.title("Sistema Inteligente de Control de Aforo - YOLO Tracking")

tk.Label(root, text="Cantidad máxima permitida:").pack()
entry = tk.Entry(root)
entry.pack()

tk.Button(root, text="Iniciar", command=iniciar).pack()

video_label = tk.Label(root)
video_label.pack()

root.mainloop()

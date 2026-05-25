import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import serial
import math
import numpy as np
from datetime import datetime

# ========================
# CARGAR MODELO YOLO
# ========================
modelp = YOLO("yolov8n.pt")
modelh = YOLO("best2.pt")

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

# Variables reutilizadas
total_personas = 0
total_cascos = 0

# ========================
# CONFIGURACIÓN VISUAL
# ========================
WIDTH = 1380
HEIGHT = 720

COLOR_PRINCIPAL = (230, 230, 230)
COLOR_SECUNDARIO = (140, 140, 140)
COLOR_ALERTA = (0, 0, 255)

# ========================
# HUD
# ========================
def dibujar_hud(frame):
    global total_personas, total_cascos

    overlay = frame.copy()

    # Fondo semitransparente
    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    # ========================
    # ESQUINAS
    # ========================
    def esquina(x, y, direction_x, direction_y, size=60):
        # Línea horizontal
        cv2.line(frame, (x, y), (x + direction_x * size, y), COLOR_PRINCIPAL, 2)
        
        # Línea vertical
        cv2.line(frame, (x, y), (x, y + direction_y * size), COLOR_PRINCIPAL, 2)

    h, w = frame.shape[:2]
    # Esquina arriba-izquierda
    esquina(20, 20, 1, 1)
    # Esquina arriba-derecha
    esquina(w - 20, 20, -1, 1)
    # Esquina abajo-izquierda
    esquina(20, h - 20, 1, -1)
    # Esquina abajo-derecha
    esquina(w - 20, h - 20, -1, -1)

    # ========================
    # TEXTO SUPERIOR
    # ========================
    cv2.putText(
        frame,
        "SISTEMA ACTIVO",
        (60, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_PRINCIPAL,
        2
    )

    cv2.putText(
        frame,
        "YOLO v8 | CV2",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_SECUNDARIO,
        2
    )

    hora_actual = datetime.now().strftime("%I:%M:%S %p").lower()

    cv2.putText(
        frame,
        hora_actual,
        (35, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_PRINCIPAL,
        2
    )

    # ========================
    # PUNTOS SUPERIORES
    # ========================
    cx = frame.shape[1] // 2

    cv2.circle(frame, (cx, 40), 10, (255, 255, 255), -1)
    cv2.circle(frame, (cx + 30, 40), 8, (100, 100, 100), -1)
    cv2.circle(frame, (cx + 60, 40), 8, (100, 100, 100), -1)

    # ========================
    # PANEL DERECHO
    # ========================
    cv2.putText(
        frame,
        str(total_personas),
        (frame.shape[1] - 70, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        1.6,
        COLOR_PRINCIPAL,
        2
    )


    cv2.putText(
        frame,
        "PERSONAS DETECTADAS",
        (1150, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_SECUNDARIO,
        2
    )


    # Escudos laterales
    for i in range(total_personas):

        y = 145 + (i * 40)

        puntos = np.array([
            [frame.shape[1] - 55, y],
            [frame.shape[1] - 45, y + 10],
            [frame.shape[1] - 48, y + 25],
            [frame.shape[1] - 55, y + 32],
            [frame.shape[1] - 62, y + 25],
            [frame.shape[1] - 65, y + 10]
        ], np.int32)

        cv2.polylines(
            frame,
            [puntos],
            True,
            COLOR_PRINCIPAL,
            1
        )

        if i < total_cascos:
            estado = "OK"
            color = (180, 255, 180)
        else:
            estado = "NO"
            color = COLOR_ALERTA

        cv2.putText(
            frame,
            estado,
            (frame.shape[1] - 95, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    
    # ========================
    # RETÍCULA CENTRAL
    # ========================
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    cv2.circle(frame, (center_x, center_y), 18, COLOR_SECUNDARIO, 2)
    cv2.circle(frame, (center_x, center_y), 5, COLOR_SECUNDARIO, -1)

    return frame

# ========================
# DETECCIÓN + TRACKING
# ========================
def detectar_personas(frame):

    global max_personas
    global total_personas
    global total_cascos

    resultadosp = modelp.track(frame, persist=True, verbose=False)
    resultadosh = modelh.track(frame, persist=True, verbose=False)

    ids_activos = set()
    helmets_detected = set()

    # PERSONAS
    for r in resultadosp:

        boxes = r.boxes

        if boxes.id is not None:

            for box, track_id in zip(boxes, boxes.id):

                clase = int(box.cls[0])

                if clase == 0:

                    track_id = int(track_id)
                    ids_activos.add(track_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"PERSONA {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

    # CASCOS
    for rh in resultadosh:

        boxesh = rh.boxes

        if boxesh.id is not None:

            for boxh, track_idh in zip(boxesh, boxesh.id):

                claseh = int(boxh.cls[0])

                if claseh == 0:

                    track_idh = int(track_idh)
                    helmets_detected.add(track_idh)
                    x1, y1, x2, y2 = map(int, boxh.xyxy[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 180),
                        2
                    )
                    cv2.putText(
                        frame,
                        "CASCO",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 180),
                        2
                    )

    total_personas = len(ids_activos)
    total_cascos = len(helmets_detected)

    # LÓGICA ORIGINAL
    if total_personas > max_personas:
        if arduino:
            arduino.write(b"A")
        else:
            print("Límite excedido - Señal A enviada (simulada)")
            print(f"Personas: {total_personas} | Cascos: {total_cascos}")
            cv2.putText(
                frame,
                "ALERTA: LIMITE DE PERSONAS EXCEDIDO",
                (frame.shape[1] // 2 - 200, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_ALERTA,
                2
            )
    elif total_personas <= max_personas:
        if total_cascos < total_personas:
            print("Alerta: Personas sin casco detectadas")
            print(f"Personas: {total_personas} | Cascos: {total_cascos}")
            cv2.putText(
                frame,
                "ALERTA: PERSONAS SIN CASCO DETECTADAS",
                (frame.shape[1] // 2 - 200, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_ALERTA,
                2
            )
        else:
            if arduino:
                arduino.write(b"B")
            else:
                print("Dentro del límite - Señal B enviada (simulada)")
                print(f"Personas: {total_personas} | Cascos: {total_cascos}")

    # HUD
    frame = dibujar_hud(frame)

    return frame

# ========================
# LOOP CÁMARA
# ========================
def update_frame():

    global cap, running

    if running:

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
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
        messagebox.showerror(
            "Error",
            "Ingrese un número válido"
        )
        return

    
    if running:
        print("El sistema ya está en funcionamiento")
        print(f"Nuevo límite de personas: {max_personas}")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        running = True
        update_frame()

# ========================
# INTERFAZ TKINTER
# ========================
root = tk.Tk()

root.title("Sistema Inteligente de Control de Aforo y Detección de Cascos")
root.configure(bg="black")

try:
    root.state('zoomed')
except:
    root.attributes('-fullscreen', True)

# Panel superior
control_frame = tk.Frame(root, bg="black")
control_frame.pack(fill="x", pady=10)

label = tk.Label(
    control_frame,
    text="Cantidad máxima permitida:",
    fg="white",
    bg="black",
    font=("Consolas", 12)
)
label.pack(side="left", padx=10)

entry = tk.Entry(
    control_frame,
    bg="#111111",
    fg="white",
    insertbackground="white",
    font=("Consolas", 12),
    width=10
)
entry.pack(side="left", padx=10)

btn = tk.Button(
    control_frame,
    text="INICIAR",
    command=iniciar,
    bg="#111111",
    fg="#1D1717",
    activebackground="#222222",
    activeforeground="white",
    relief="flat",
    font=("Consolas", 11, "bold")
)
btn.pack(side="left", padx=10)

# Video
video_label = tk.Label(
    root,
    bg="black",
    borderwidth=0,
    highlightthickness=0
)
video_label.pack(expand=True)

root.mainloop()

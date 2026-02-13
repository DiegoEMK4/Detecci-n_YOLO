import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import queue

cap = None
reproduciendo = False
usando_camara = False
video_fps = 30
modelo_cargado = False
resultados_queue = queue.Queue(maxsize=1)

# Cargar el modelo YOLO en el hilo principal al inicio
model = YOLO('C:/Users/diego/Desktop/plecostomus_project/plecostomus-seg/yolov8n-seg-custom3/weights/best.pt')
modelo_cargado = True

def procesar_fotogramas():
    global cap, reproduciendo, usando_camara, video_fps, model, resultados_queue
    while reproduciendo and cap and cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            results = model(frame)
            # Colocar tanto los resultados como el fotograma original en la cola
            resultados_queue.put((results, frame))
            processing_time = time.time() - start_time
            delay = max(1, int((1/video_fps - processing_time) * 1000))
            time.sleep(delay / 1000)
        else:
            if not usando_camara:
                cap.release()
                reproduciendo = False
            break

def actualizar_video():
    global reproduciendo, resultados_queue
    if reproduciendo:
        try:
            results, frame = resultados_queue.get_nowait()
            if frame is not None:
                annotated_frame = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil.resize((int(ancho_pantalla * 0.8), int(alto_pantalla * 0.6))))
                canvas_video.create_image(0, 0, anchor=tk.NW, image=frame_tk)
                canvas_video.image = frame_tk
        except queue.Empty:
            pass

        ventana.after(int(1000 / video_fps), actualizar_video)

def iniciar_reproduccion():
    global reproduciendo, procesamiento_thread
    reproduciendo = True
    procesamiento_thread = threading.Thread(target=procesar_fotogramas)
    procesamiento_thread.daemon = True
    procesamiento_thread.start()
    actualizar_video()

def cargar_video():
    global cap, reproduciendo, usando_camara, video_fps
    ruta_video = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi;*.mov")])
    if ruta_video:
        if cap and cap.isOpened():
            cap.release()
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            print("Error al abrir el video")
            return
        reproduciendo = False
        usando_camara = False
        video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        iniciar_reproduccion()

def habilitar_camara():
    global cap, reproduciendo, usando_camara, video_fps
    if cap and cap.isOpened():
        cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara")
        return
    reproduciendo = False
    usando_camara = True
    video_fps = 30
    iniciar_reproduccion()

def reproducir_video():
    global reproduciendo, procesamiento_thread
    if cap and cap.isOpened() and not reproduciendo:
        iniciar_reproduccion()
    elif reproduciendo:
        pass

def pausar_video():
    global reproduciendo
    reproduciendo = not reproduciendo

def salir():
    global cap, reproduciendo
    reproduciendo = False
    if cap and cap.isOpened():
        cap.release()
    ventana.destroy()

ventana = tk.Tk()
ventana.title("Software para detectar el pez plecostomus")
ancho_pantalla = ventana.winfo_screenwidth()
alto_pantalla = ventana.winfo_screenheight()
ventana.geometry(f"{ancho_pantalla}x{alto_pantalla}")
ventana.configure(bg="lightgray")
titulo = tk.Label(ventana, text="Detección del Pez Plecostomus", font=("Helvetica", 24, "bold"), bg="lightgray")
titulo.pack(pady=20)
canvas_video = tk.Canvas(ventana, width=ancho_pantalla * 0.8, height=alto_pantalla * 0.6, bg="black")
canvas_video.pack(pady=20)
marco_botones = tk.Frame(ventana, bg="lightgray")
marco_botones.pack(pady=20)
boton_cargar = tk.Button(marco_botones, text="Cargar Video", command=cargar_video, padx=20, pady=10)
boton_camara = tk.Button(marco_botones, text="Habilitar Cámara", command=habilitar_camara, padx=20, pady=10)
boton_reproducir = tk.Button(marco_botones, text="Reproducir", command=reproducir_video, padx=20, pady=10)
boton_pausar = tk.Button(marco_botones, text="Pausar Video", command=pausar_video, padx=20, pady=10)
boton_salir = tk.Button(marco_botones, text="Salir", command=salir, padx=20, pady=10)
boton_cargar.pack(side=tk.LEFT, padx=5)
boton_camara.pack(side=tk.LEFT, padx=5)
boton_reproducir.pack(side=tk.LEFT, padx=5)
boton_pausar.pack(side=tk.LEFT, padx=5)
boton_salir.pack(side=tk.LEFT, padx=5)

ventana.mainloop()
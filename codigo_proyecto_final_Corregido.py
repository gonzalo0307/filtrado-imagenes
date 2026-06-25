import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
import threading


# ══════════════════════════════════════════════════════════════
#  FUNCIONES DE FILTRADO
# ══════════════════════════════════════════════════════════════

def AplicarConvolucion(MatrizEntrada, KernelConvolucion):
    AlturaKernel, AnchoKernel = KernelConvolucion.shape
    PaddingAltura = AlturaKernel // 2
    PaddingAncho  = AnchoKernel  // 2
    MatrizPadded  = np.pad(
        MatrizEntrada.astype(float),
        ((PaddingAltura, PaddingAltura), (PaddingAncho, PaddingAncho)),
        mode='reflect'
    )
    
    MatrizSalida = np.zeros_like(MatrizEntrada, dtype=float)
    for FilaActual in range(MatrizEntrada.shape[0]):
        for ColumnaActual in range(MatrizEntrada.shape[1]):
            RegionVecinos = MatrizPadded[
                FilaActual:FilaActual + AlturaKernel,
                ColumnaActual:ColumnaActual + AnchoKernel
            ]
            MatrizSalida[FilaActual, ColumnaActual] = np.sum(RegionVecinos * KernelConvolucion)
    return MatrizSalida


def AplicarConvolucionRapida(MatrizEntrada, KernelConvolucion):
    """Convolución vectorizada para imágenes grandes (mucho más rápida)."""
    from numpy.lib.stride_tricks import sliding_window_view
    AlturaKernel, AnchoKernel = KernelConvolucion.shape
    PaddingAltura = AlturaKernel // 2
    PaddingAncho  = AnchoKernel  // 2
    MatrizPadded  = np.pad(
        MatrizEntrada.astype(float),
        ((PaddingAltura, PaddingAltura), (PaddingAncho, PaddingAncho)),
        mode='reflect'
    )
    Ventanas = sliding_window_view(MatrizPadded, (AlturaKernel, AnchoKernel))
    return (Ventanas * KernelConvolucion).sum(axis=(-2, -1))


def AplicarFiltroMedia(MatrizPixeles, Rapido=False):
    KernelMedia = np.ones((3, 3)) / 9.0
    Fn = AplicarConvolucionRapida if Rapido else AplicarConvolucion
    return np.clip(Fn(MatrizPixeles, KernelMedia), 0, 255).astype(np.uint8)
 

def AplicarFiltroMediana(MatrizPixeles, Rapido=False):
    if Rapido:
        from numpy.lib.stride_tricks import sliding_window_view
        MatrizPadded = np.pad(MatrizPixeles.astype(float), 1, mode='reflect')
        Ventanas = sliding_window_view(MatrizPadded, (3, 3))
        return np.median(Ventanas, axis=(-2, -1)).astype(np.uint8)
    MatrizPadded = np.pad(MatrizPixeles.astype(float), 1, mode='reflect')
    MatrizSalida = np.zeros_like(MatrizPixeles, dtype=float)
    for FilaActual in range(MatrizPixeles.shape[0]):
        for ColumnaActual in range(MatrizPixeles.shape[1]):
            VentanaVecinos = MatrizPadded[
                FilaActual:FilaActual + 3,
                ColumnaActual:ColumnaActual + 3
            ].flatten()
            MatrizSalida[FilaActual, ColumnaActual] = np.median(VentanaVecinos)
    return np.clip(MatrizSalida, 0, 255).astype(np.uint8)


def AplicarFiltroLaplaciano(MatrizPixeles, Rapido=False):
    KernelLaplaciano = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=float)
    Fn = AplicarConvolucionRapida if Rapido else AplicarConvolucion
    return Fn(MatrizPixeles, KernelLaplaciano)


def AplicarFiltroSobel(MatrizPixeles, Rapido=False):
    KernelSobelH = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    KernelSobelV = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    Fn = AplicarConvolucionRapida if Rapido else AplicarConvolucion
    GH = Fn(MatrizPixeles, KernelSobelH)
    GV = Fn(MatrizPixeles, KernelSobelV)
    return np.sqrt(GH**2 + GV**2)


def ReescalarMatrizA255(MatrizSinRecortar):
    ValorMinimo = MatrizSinRecortar.min()
    ValorMaximo = MatrizSinRecortar.max()
    if ValorMaximo == ValorMinimo:
        return np.zeros_like(MatrizSinRecortar, dtype=np.uint8)
    return ((MatrizSinRecortar - ValorMinimo) / (ValorMaximo - ValorMinimo) * 255).astype(np.uint8)


def ConvertirMatrizATexto(MatrizNumerica, UsarDecimales=False):
    FilasTexto = []
    for FilaActual in MatrizNumerica:
        if UsarDecimales:
            FilasTexto.append("    ".join(f"{Valor:9.2f}" for Valor in FilaActual))
        else:
            FilasTexto.append("    ".join(f"{int(Valor):3d}" for Valor in FilaActual))
    return "\n".join(FilasTexto)


# ══════════════════════════════════════════════════════════════
#  GENERADOR DE IMAGEN DIGITALIZADA (solo para 15x15)
# ══════════════════════════════════════════════════════════════

def CrearImagenDigitalizada(MatrizUint8, TamanoCelda=30):
    CantidadFilas, CantidadColumnas = MatrizUint8.shape
    AnchoCantvas  = CantidadColumnas * TamanoCelda
    AlturaCantvas = CantidadFilas    * TamanoCelda
    ImagenCanvas  = Image.new("RGB", (AnchoCantvas, AlturaCantvas), "#0a0a14")
    ObjetoDibujo  = ImageDraw.Draw(ImagenCanvas)
    for FilaActual in range(CantidadFilas):
        for ColumnaActual in range(CantidadColumnas):
            ValorPixel = int(MatrizUint8[FilaActual, ColumnaActual])
            X0 = ColumnaActual * TamanoCelda
            Y0 = FilaActual    * TamanoCelda
            X1 = X0 + TamanoCelda - 1
            Y1 = Y0 + TamanoCelda - 1
            ObjetoDibujo.rectangle([X0, Y0, X1, Y1],
                                    fill=(ValorPixel, ValorPixel, ValorPixel))
            ObjetoDibujo.rectangle([X0, Y0, X1, Y1],
                                    outline="#7c3aed", width=1)
            ColorTexto = "white" if ValorPixel < 128 else "black"
            TextoValor = str(ValorPixel)
            PosicionX  = X0 + TamanoCelda // 2 - len(TextoValor) * 3
            PosicionY  = Y0 + TamanoCelda // 2 - 5
            ObjetoDibujo.text((PosicionX, PosicionY), TextoValor, fill=ColorTexto)
    return ImagenCanvas


# ══════════════════════════════════════════════════════════════
#  PALETA DE COLORES
# ══════════════════════════════════════════════════════════════

C = {
    "Fondo":           "#1e1e2e",
    "Panel":           "#2a2a3e",
    "Acento":          "#7c3aed",
    "AcentoClaro":     "#9d5cf5",
    "TextoPrincipal":  "#e2e8f0",
    "TextoSecundario": "#94a3b8",
    "Verde":           "#10b981",
    "Rojo":            "#ef4444",
    "Naranja":         "#f59e0b",
    "Negro":           "#0f0f1a",
    "AzulOscuro":      "#0a0a14",
    "Cyan":            "#a5f3fc",
}


# ══════════════════════════════════════════════════════════════
#  APLICACION PRINCIPAL
# ══════════════════════════════════════════════════════════════

class AplicacionFiltrado(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Filtrado de Imágenes — Versión Completa")
        self.configure(bg=C["Fondo"])
        self.geometry("1500x900")
        self.resizable(True, True)

        # Estado
        self.ImagenOriginalCargada  = None   # PIL Image en grises
        self.ImagenColorOriginal    = None   # PIL Image en color (RGB) si aplica
        self.EsImagenColor          = False
        self.ImagenRecortada15x15   = None
        self.MatrizRecortada15x15   = None
        self.MatrizImagenCompleta   = None   # grises, imagen completa

        self.ConstruirInterfaz()

    # ─────────────────────────────────────────────────────────
    #  INTERFAZ
    # ─────────────────────────────────────────────────────────

    def ConstruirInterfaz(self):
        self.ConstruirSidebar()
        self.ConstruirAreaPrincipal()

    def ConstruirSidebar(self):
        SB = tk.Frame(self, bg=C["Panel"], width=285)
        SB.pack(side="left", fill="y")
        SB.pack_propagate(False)

        # Título
        tk.Label(SB, text="FILTRADO\nDE IMÁGENES",
                 bg=C["Panel"], fg=C["Acento"],
                 font=("Courier New", 13, "bold"), justify="center").pack(pady=(22, 2))
        tk.Label(SB, text="Procesamiento digital · Color & Grises",
                 bg=C["Panel"], fg=C["TextoSecundario"],
                 font=("Courier New", 7), wraplength=240).pack(pady=(0, 10))
        tk.Frame(SB, bg=C["Acento"], height=2).pack(fill="x", padx=18, pady=4)

        # ── PASO 1 ──
        self._SeccionLabel(SB, "PASO 1 — CARGAR IMAGEN")
        tk.Label(SB, text="JPG/JPEG · color o blanco y negro",
                 bg=C["Panel"], fg=C["TextoSecundario"],
                 font=("Courier New", 7), wraplength=240).pack(padx=18, anchor="w")
        tk.Button(SB, text="Abrir imagen JPG / JPEG",
                  bg=C["Acento"], fg="white",
                  font=("Courier New", 9, "bold"), relief="flat",
                  cursor="hand2", pady=7,
                  command=self.AccionCargarImagen).pack(padx=18, pady=(6, 4), fill="x")

        self.EtiquetaInfoImagen = tk.Label(SB, text="Sin imagen cargada",
                                           bg=C["Panel"], fg=C["TextoSecundario"],
                                           font=("Courier New", 7), wraplength=240)
        self.EtiquetaInfoImagen.pack(pady=2)

        # Badge modo
        self.BadgeModo = tk.Label(SB, text="",
                                   bg=C["Panel"], fg=C["Verde"],
                                   font=("Courier New", 8, "bold"))
        self.BadgeModo.pack(pady=2)

        tk.Frame(SB, bg="#3a3a5e", height=1).pack(fill="x", padx=18, pady=8)

        # ── PASO 2 ──
        self._SeccionLabel(SB, "PASO 2 — RECORTE 15×15")
        tk.Label(SB, text="Elige desde donde iniciar el recorte\n(X = columna,  Y = fila)",
                 bg=C["Panel"], fg=C["TextoSecundario"],
                 font=("Courier New", 7), wraplength=240, justify="left").pack(padx=18, anchor="w")

        CoordFrame = tk.Frame(SB, bg=C["Panel"])
        CoordFrame.pack(padx=18, fill="x", pady=4)
        for Idx, (Lbl, Attr) in enumerate([("X inicio:", "EntradaCoordenadaX"),
                                            ("Y inicio:", "EntradaCoordenadaY")]):
            tk.Label(CoordFrame, text=Lbl, bg=C["Panel"], fg=C["TextoSecundario"],
                     font=("Courier New", 8), width=9, anchor="w").grid(row=Idx, column=0, pady=3)
            E = tk.Entry(CoordFrame, width=7, bg=C["Negro"], fg=C["TextoPrincipal"],
                         insertbackground=C["TextoPrincipal"],
                         font=("Courier New", 10), relief="flat", bd=4)
            E.insert(0, "0")
            E.grid(row=Idx, column=1, pady=3, padx=4)
            setattr(self, Attr, E)

        tk.Button(SB, text="Aplicar recorte 15×15",
                  bg=C["Negro"], fg=C["Verde"],
                  font=("Courier New", 9, "bold"), relief="flat",
                  cursor="hand2", pady=7,
                  command=self.AccionAplicarRecorte).pack(padx=18, pady=(6, 2), fill="x")

        tk.Frame(SB, bg="#3a3a5e", height=1).pack(fill="x", padx=18, pady=8)

        # ── PASO 3 ──
        self._SeccionLabel(SB, "PASO 3 — FILTRO")
        self.VariableFiltro = tk.StringVar(value="media")
        for Texto, Val in [("Media  (suaviza)",   "media"),
                            ("Mediana  (ruido)",   "mediana"),
                            ("Laplaciano  (bordes)", "laplaciano"),
                            ("Sobel  (bordes)",    "sobel")]:
            tk.Radiobutton(SB, text=f" {Texto}", variable=self.VariableFiltro, value=Val,
                           bg=C["Panel"], fg=C["TextoPrincipal"],
                           selectcolor=C["Acento"],
                           activebackground=C["Panel"],
                           activeforeground=C["TextoPrincipal"],
                           font=("Courier New", 8)).pack(anchor="w", padx=24, pady=2)

        tk.Frame(SB, bg="#3a3a5e", height=1).pack(fill="x", padx=18, pady=6)

        # Selector de modo
        self._SeccionLabel(SB, "MODO DE PROCESAMIENTO")
        self.VariableModo = tk.StringVar(value="recorte")
        for Texto, Val in [("Solo recorte 15×15",   "recorte"),
                            ("Imagen completa",      "completa"),
                            ("Ambos",                "ambos")]:
            tk.Radiobutton(SB, text=f" {Texto}", variable=self.VariableModo, value=Val,
                           bg=C["Panel"], fg=C["TextoPrincipal"],
                           selectcolor=C["AcentoClaro"],
                           activebackground=C["Panel"],
                           activeforeground=C["TextoPrincipal"],
                           font=("Courier New", 8)).pack(anchor="w", padx=24, pady=2)

        tk.Button(SB, text="▶  Aplicar filtro",
                  bg=C["Acento"], fg="white",
                  font=("Courier New", 10, "bold"), relief="flat",
                  cursor="hand2", pady=9,
                  command=self.AccionAplicarFiltro).pack(padx=18, pady=(10, 6), fill="x")

        # Barra de estado
        self.EtiquetaEstado = tk.Label(self,
                                       text="  Listo. Sigue los pasos del panel izquierdo.",
                                       bg=C["Negro"], fg=C["TextoSecundario"],
                                       font=("Courier New", 8), anchor="w")
        self.EtiquetaEstado.pack(side="bottom", fill="x", ipady=3)

    def ConstruirAreaPrincipal(self):
        ContenedorPrincipal = tk.Frame(self, bg=C["Fondo"])
        ContenedorPrincipal.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        Estilo = ttk.Style(self)
        Estilo.theme_use("clam")
        Estilo.configure("TNotebook", background=C["Fondo"], borderwidth=0)
        Estilo.configure("TNotebook.Tab", background=C["Panel"],
                          foreground=C["TextoSecundario"],
                          padding=[10, 5], font=("Courier New", 8, "bold"))
        Estilo.map("TNotebook.Tab",
                   background=[("selected", C["Acento"])],
                   foreground=[("selected", "white")])
        Estilo.configure("TFrame", background=C["Fondo"])

        self.NB = ttk.Notebook(ContenedorPrincipal)
        self.NB.pack(fill="both", expand=True)

        # ── Pestaña 0: Original ──
        P0 = ttk.Frame(self.NB)
        self.NB.add(P0, text="  Original  ")
        self._ConstruirPestanaOriginal(P0)

        # ── Pestaña 1: Recorte 15x15 ──
        P1 = ttk.Frame(self.NB)
        self.NB.add(P1, text="  Recorte 15×15  ")
        self._ConstruirPestanaRecorte(P1)

        # ── Pestaña 2: Media / Mediana — Recorte ──
        P2 = ttk.Frame(self.NB)
        self.NB.add(P2, text="  Media/Mediana · Recorte  ")
        self._ConstruirPestanaMediaMedianaRecorte(P2)

        # ── Pestaña 3: Laplaciano / Sobel — Recorte ──
        P3 = ttk.Frame(self.NB)
        self.NB.add(P3, text="  Laplaciano/Sobel · Recorte  ")
        self._ConstruirPestanaLaplacianoSobelRecorte(P3)

        # ── Pestaña 4: Media / Mediana — Completa ──
        P4 = ttk.Frame(self.NB)
        self.NB.add(P4, text="  Media/Mediana · Completa  ")
        self._ConstruirPestanaMediaMedianaCompleta(P4)

        # ── Pestaña 5: Laplaciano / Sobel — Completa ──
        P5 = ttk.Frame(self.NB)
        self.NB.add(P5, text="  Laplaciano/Sobel · Completa  ")
        self._ConstruirPestanaLaplacianoSobelCompleta(P5)

    # ─────────────────────────────────────────────────────────
    #  CONSTRUCCIÓN DE PESTAÑAS
    # ─────────────────────────────────────────────────────────

    def _ConstruirPestanaOriginal(self, P):
        Marco = tk.Frame(P, bg=C["Fondo"])
        Marco.pack(fill="both", expand=True, padx=8, pady=8)

        # Fila superior: imagen original color + grises
        FilaSup = tk.Frame(Marco, bg=C["Fondo"], height=380)
        FilaSup.pack(fill="x", pady=(0, 6))
        FilaSup.pack_propagate(False)

        self.LabelImgOriginalColor  = self._ContenedorImagen(FilaSup, "Carga una imagen JPG/JPEG", "Imagen original")
        self.LabelImgOriginalGrises = self._ContenedorImagen(FilaSup, "—", "Versión en grises (para filtros)")

        # Info
        self.LabelInfoOriginal = tk.Label(Marco,
                                           text="Carga una imagen para comenzar.",
                                           bg=C["Negro"], fg=C["TextoSecundario"],
                                           font=("Courier New", 8), pady=6)
        self.LabelInfoOriginal.pack(fill="x", padx=4)

    def _ConstruirPestanaRecorte(self, P):
        FilaSup = tk.Frame(P, bg=C["Fondo"], height=310)
        FilaSup.pack(fill="x", padx=8, pady=(8, 4))
        FilaSup.pack_propagate(False)

        self.LabelRecorteAmpliado     = self._ContenedorImagen(FilaSup, "Recorte ampliado (zoom)", "Imagen recortada")
        self.LabelDigitalizadaRecorte = self._ContenedorImagen(FilaSup, "Cada pixel con su valor", "Imagen digitalizada")

        FilaInf = tk.Frame(P, bg=C["Fondo"])
        FilaInf.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.TextoMatrizRecorte = self._CajaTexto(FilaInf, "Matriz numérica del recorte (valores 0–255):")

    def _ConstruirPestanaMediaMedianaRecorte(self, P):
        FilaSup = tk.Frame(P, bg=C["Fondo"], height=290)
        FilaSup.pack(fill="x", padx=8, pady=(8, 4))
        FilaSup.pack_propagate(False)

        self.LabelRecMM_IniImg  = self._ContenedorImagen(FilaSup, "—", "Imagen inicial")
        self.LabelRecMM_IniDig  = self._ContenedorImagen(FilaSup, "—", "Digitalización inicial")
        self.LabelRecMM_ResImg  = self._ContenedorImagen(FilaSup, "—", "Imagen resultado")
        self.LabelRecMM_ResDig  = self._ContenedorImagen(FilaSup, "—", "Digitalización resultado")

        FilaInf = tk.Frame(P, bg=C["Fondo"])
        FilaInf.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.TextoRecMM_MatIni = self._CajaTexto(FilaInf, "Matriz inicial (0–255):")
        self.TextoRecMM_MatRes = self._CajaTexto(FilaInf, "Matriz resultado tras aplicar el filtro (0–255):")

    def _ConstruirPestanaLaplacianoSobelRecorte(self, P):
        FilaSup = tk.Frame(P, bg=C["Fondo"], height=290)
        FilaSup.pack(fill="x", padx=8, pady=(8, 4))
        FilaSup.pack_propagate(False)

        self.LabelRecLS_IniImg  = self._ContenedorImagen(FilaSup, "—", "Imagen inicial")
        self.LabelRecLS_IniDig  = self._ContenedorImagen(FilaSup, "—", "Digitalización inicial")
        self.LabelRecLS_ResImg  = self._ContenedorImagen(FilaSup, "—", "Imagen resultado (re-escalada)")
        self.LabelRecLS_ResDig  = self._ContenedorImagen(FilaSup, "—", "Digitalización resultado")

        FilaInf = tk.Frame(P, bg=C["Fondo"])
        FilaInf.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.TextoRecLS_MatSin = self._CajaTexto(FilaInf, "Matriz resultado sin re-escalar:")
        self.TextoRecLS_MatRes = self._CajaTexto(FilaInf, "Matriz re-escalada (0–255):")

    def _ConstruirPestanaMediaMedianaCompleta(self, P):
        FilaSup = tk.Frame(P, bg=C["Fondo"], height=340)
        FilaSup.pack(fill="x", padx=8, pady=(8, 4))
        FilaSup.pack_propagate(False)

        self.LabelCompMM_IniImg = self._ContenedorImagen(FilaSup, "—", "Imagen inicial (grises)")
        self.LabelCompMM_ResImg = self._ContenedorImagen(FilaSup, "—", "Imagen resultado")

        # Barra de progreso
        BarraFrame = tk.Frame(P, bg=C["Fondo"])
        BarraFrame.pack(fill="x", padx=8)
        self.LabelProgCompMM = tk.Label(BarraFrame, text="",
                                         bg=C["Fondo"], fg=C["Naranja"],
                                         font=("Courier New", 8))
        self.LabelProgCompMM.pack(anchor="w")

        FilaInf = tk.Frame(P, bg=C["Fondo"])
        FilaInf.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Info matriz (resumida para imágenes grandes)
        self.TextoCompMM_Info = self._CajaTexto(FilaInf, "Información de la matriz resultado:")

    def _ConstruirPestanaLaplacianoSobelCompleta(self, P):
        FilaSup = tk.Frame(P, bg=C["Fondo"], height=340)
        FilaSup.pack(fill="x", padx=8, pady=(8, 4))
        FilaSup.pack_propagate(False)

        self.LabelCompLS_IniImg = self._ContenedorImagen(FilaSup, "—", "Imagen inicial (grises)")
        self.LabelCompLS_ResImg = self._ContenedorImagen(FilaSup, "—", "Imagen resultado (re-escalada)")

        BarraFrame = tk.Frame(P, bg=C["Fondo"])
        BarraFrame.pack(fill="x", padx=8)
        self.LabelProgCompLS = tk.Label(BarraFrame, text="",
                                         bg=C["Fondo"], fg=C["Naranja"],
                                         font=("Courier New", 8))
        self.LabelProgCompLS.pack(anchor="w")

        FilaInf = tk.Frame(P, bg=C["Fondo"])
        FilaInf.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.TextoCompLS_Info = self._CajaTexto(FilaInf, "Información de la matriz resultado:")

    # ─────────────────────────────────────────────────────────
    #  WIDGETS AUXILIARES
    # ─────────────────────────────────────────────────────────

    def _SeccionLabel(self, Padre, Texto):
        tk.Label(Padre, text=Texto, bg=C["Panel"], fg=C["TextoPrincipal"],
                 font=("Courier New", 8, "bold")).pack(pady=(6, 2), padx=18, anchor="w")

    def _ContenedorImagen(self, Padre, Placeholder, Titulo=None):
        Marco = tk.Frame(Padre, bg=C["Negro"])
        Marco.pack(side="left", padx=5, pady=4, fill="both", expand=True)
        if Titulo:
            tk.Label(Marco, text=Titulo, bg=C["Negro"], fg=C["Acento"],
                     font=("Courier New", 8, "bold")).pack(pady=(6, 2))
        Lbl = tk.Label(Marco, text=Placeholder, bg=C["Negro"], fg="#3a3a5e",
                        font=("Courier New", 8), wraplength=220)
        Lbl.pack(expand=True, pady=4)
        return Lbl

    def _CajaTexto(self, Padre, Titulo):
        Marco = tk.Frame(Padre, bg=C["Negro"])
        Marco.pack(side="left", padx=5, pady=4, fill="both", expand=True)
        tk.Label(Marco, text=Titulo, bg=C["Negro"], fg=C["Acento"],
                 font=("Courier New", 8, "bold"), anchor="w").pack(fill="x", padx=6, pady=(6, 2))
        CT = tk.Text(Marco, bg=C["AzulOscuro"], fg=C["Cyan"],
                      font=("Courier New", 9), relief="flat",
                      state="disabled", wrap="none")
        SV = tk.Scrollbar(Marco, command=CT.yview)
        SH = tk.Scrollbar(Marco, orient="horizontal", command=CT.xview)
        CT.configure(yscrollcommand=SV.set, xscrollcommand=SH.set)
        SV.pack(side="right", fill="y")
        SH.pack(side="bottom", fill="x")
        CT.pack(fill="both", expand=True, padx=6, pady=6)
        return CT

    def _MostrarImagen(self, Label, ImagenPIL, MaxTam):
        Copia = ImagenPIL.copy()
        Copia.thumbnail((MaxTam, MaxTam), Image.LANCZOS)
        Foto = ImageTk.PhotoImage(Copia)
        Label.config(image=Foto, text="", bg=C["Negro"])
        Label.image = Foto

    def _EscribirTexto(self, Widget, Texto):
        Widget.config(state="normal")
        Widget.delete("1.0", "end")
        Widget.insert("1.0", Texto)
        Widget.config(state="disabled")

    def _Estado(self, Msg, Color=None):
        self.EtiquetaEstado.config(text=f"  {Msg}",
                                    fg=Color or C["TextoSecundario"])

    # ─────────────────────────────────────────────────────────
    #  ACCIÓN: CARGAR IMAGEN
    # ─────────────────────────────────────────────────────────

    def AccionCargarImagen(self):
        Ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes JPEG", "*.jpg *.jpeg"), ("Todos los archivos", "*.*")]
        )
        if not Ruta:
            return

        Ext = os.path.splitext(Ruta)[1].lower()
        if Ext not in (".jpg", ".jpeg"):
            messagebox.showerror("Formato incorrecto",
                                  "Solo se aceptan archivos JPG o JPEG.")
            return

        ImgRGB  = Image.open(Ruta).convert("RGB")
        ArrayRGB = np.array(ImgRGB)

        CanalRigualG = np.array_equal(ArrayRGB[:, :, 0], ArrayRGB[:, :, 1])
        CanalGigualB = np.array_equal(ArrayRGB[:, :, 1], ArrayRGB[:, :, 2])
        EsGris = CanalRigualG and CanalGigualB

        if EsGris:
            self.EsImagenColor       = False
            self.ImagenColorOriginal = None
            self.ImagenOriginalCargada = ImgRGB.convert("L")
            BadgeTexto = " Imagen en escala de grises"
            BadgeColor = C["Verde"]
        else:
            self.EsImagenColor         = True
            self.ImagenColorOriginal   = ImgRGB
            self.ImagenOriginalCargada = ImgRGB.convert("L")
            BadgeTexto = " Imagen a COLOR -> convertida \n a grises para filtros"
            BadgeColor = C["Naranja"]

        self.MatrizImagenCompleta  = np.array(self.ImagenOriginalCargada)
        self.ImagenRecortada15x15  = None
        self.MatrizRecortada15x15  = None

        # Mostrar en pestaña Original
        if self.EsImagenColor:
            self._MostrarImagen(self.LabelImgOriginalColor,  self.ImagenColorOriginal,    560)
        else:
            self._MostrarImagen(self.LabelImgOriginalColor,  self.ImagenOriginalCargada,  560)
        self._MostrarImagen(self.LabelImgOriginalGrises, self.ImagenOriginalCargada, 560)

        Nombre = os.path.basename(Ruta)
        W, H   = self.ImagenOriginalCargada.size
        self.EtiquetaInfoImagen.config(
            text=f" {Nombre}\n{W}×{H} px")
        self.BadgeModo.config(text=BadgeTexto, fg=BadgeColor)

        ModoStr = "color" if self.EsImagenColor else "grises"
        self.LabelInfoOriginal.config(
            text=f"Imagen cargada: {Nombre}  |  {W}×{H} px  |  Modo: {ModoStr}  "
                 f"|  {'Convertida a grises para filtros' if self.EsImagenColor else 'Lista para filtros'}")

        self._Estado(f"Imagen cargada: {Nombre}  ({W}×{H})", C["Verde"])
        self.NB.select(0)

    # ─────────────────────────────────────────────────────────
    #  ACCIÓN: RECORTE 15×15
    # ─────────────────────────────────────────────────────────

    def AccionAplicarRecorte(self):
        if self.ImagenOriginalCargada is None:
            messagebox.showwarning("Sin imagen", "Primero carga una imagen.")
            return
        try:
            CX = int(self.EntradaCoordenadaX.get())
            CY = int(self.EntradaCoordenadaY.get())
        except ValueError:
            messagebox.showerror("Error", "X e Y deben ser enteros.")
            return

        W, H = self.ImagenOriginalCargada.size
        if CX < 0 or CY < 0 or CX + 15 > W or CY + 15 > H:
            messagebox.showerror("Fuera de rango",
                                  f"No cabe un recorte 15×15 desde ({CX},{CY}).\n"
                                  f"La imagen mide {W}×{H} px.")
            return

        self.ImagenRecortada15x15 = self.ImagenOriginalCargada.crop(
            (CX, CY, CX + 15, CY + 15))
        self.MatrizRecortada15x15 = np.array(self.ImagenRecortada15x15)

        ImgZoom = self.ImagenRecortada15x15.resize((240, 240), Image.NEAREST)
        ImgDig  = CrearImagenDigitalizada(self.MatrizRecortada15x15, TamanoCelda=16)

        self._MostrarImagen(self.LabelRecorteAmpliado,     ImgZoom, 260)
        self._MostrarImagen(self.LabelDigitalizadaRecorte, ImgDig,  260)
        self._EscribirTexto(self.TextoMatrizRecorte,
                             ConvertirMatrizATexto(self.MatrizRecortada15x15))

        self._Estado(f"Recorte 15×15 aplicado desde ({CX},{CY}).", C["Verde"])
        self.NB.select(1)

    # ─────────────────────────────────────────────────────────
    #  ACCIÓN: APLICAR FILTRO
    # ─────────────────────────────────────────────────────────

    def AccionAplicarFiltro(self):
        Filtro = self.VariableFiltro.get()
        Modo   = self.VariableModo.get()

        HacerRecorte  = Modo in ("recorte", "ambos")
        HacerCompleta = Modo in ("completa", "ambos")

        if HacerRecorte and self.MatrizRecortada15x15 is None:
            messagebox.showwarning("Sin recorte",
                                    "Primero aplica el recorte 15×15 (Paso 2).")
            return
        if HacerCompleta and self.MatrizImagenCompleta is None:
            messagebox.showwarning("Sin imagen", "Primero carga una imagen.")
            return

        # ── Procesar recorte ──
        if HacerRecorte:
            self._ProcesarRecorte(Filtro)

        # ── Procesar imagen completa en hilo separado ──
        if HacerCompleta:
            self._Estado("Procesando imagen completa… por favor espera.", C["Naranja"])
            self.update()
            hilo = threading.Thread(target=self._ProcesarCompleta,
                                     args=(Filtro,), daemon=True)
            hilo.start()
        else:
            self._Estado(f"Filtro '{Filtro}' aplicado al recorte.", C["Verde"])

    def _ProcesarRecorte(self, Filtro):
        MatrizEnt  = self.MatrizRecortada15x15.copy()
        ImgZoomOri = self.ImagenRecortada15x15.resize((240, 240), Image.NEAREST)
        DigOri     = CrearImagenDigitalizada(MatrizEnt, TamanoCelda=16)

        if Filtro in ("media", "mediana"):
            MatrizRes = (AplicarFiltroMedia(MatrizEnt)
                         if Filtro == "media"
                         else AplicarFiltroMediana(MatrizEnt))
            ImgRes  = Image.fromarray(MatrizRes, "L")
            ImgZoom = ImgRes.resize((240, 240), Image.NEAREST)
            DigRes  = CrearImagenDigitalizada(MatrizRes, TamanoCelda=16)

            self._MostrarImagen(self.LabelRecMM_IniImg, ImgZoomOri, 240)
            self._MostrarImagen(self.LabelRecMM_IniDig, DigOri,     260)
            self._MostrarImagen(self.LabelRecMM_ResImg, ImgZoom,    240)
            self._MostrarImagen(self.LabelRecMM_ResDig, DigRes,     260)
            self._EscribirTexto(self.TextoRecMM_MatIni, ConvertirMatrizATexto(MatrizEnt))
            self._EscribirTexto(self.TextoRecMM_MatRes, ConvertirMatrizATexto(MatrizRes))
            self.NB.select(2)
        else:
            MatrizSin = (AplicarFiltroLaplaciano(MatrizEnt)
                         if Filtro == "laplaciano"
                         else AplicarFiltroSobel(MatrizEnt))
            MatrizRe  = ReescalarMatrizA255(MatrizSin)
            ImgRes    = Image.fromarray(MatrizRe, "L")
            ImgZoom   = ImgRes.resize((240, 240), Image.NEAREST)
            DigRes    = CrearImagenDigitalizada(MatrizRe, TamanoCelda=16)

            self._MostrarImagen(self.LabelRecLS_IniImg, ImgZoomOri, 240)
            self._MostrarImagen(self.LabelRecLS_IniDig, DigOri,     260)
            self._MostrarImagen(self.LabelRecLS_ResImg, ImgZoom,    240)
            self._MostrarImagen(self.LabelRecLS_ResDig, DigRes,     260)
            self._EscribirTexto(self.TextoRecLS_MatSin,
                                 ConvertirMatrizATexto(MatrizSin, UsarDecimales=True))
            self._EscribirTexto(self.TextoRecLS_MatRes, ConvertirMatrizATexto(MatrizRe))
            self.NB.select(3)

    def _ProcesarCompleta(self, Filtro):
        """Corre en hilo separado para no bloquear la UI."""
        try:
            MatrizEnt = self.MatrizImagenCompleta.copy()
            H, W      = MatrizEnt.shape

            # Mostrar imagen inicial
            ImgIni = Image.fromarray(MatrizEnt, "L")

            if Filtro in ("media", "mediana"):
                LblProg = self.LabelProgCompMM
                LblProg.config(text=f"⏳ Aplicando filtro {Filtro} a imagen {W}×{H}…")
                self.update_idletasks()

                MatrizRes = (AplicarFiltroMedia(MatrizEnt, Rapido=True)
                             if Filtro == "media"
                             else AplicarFiltroMediana(MatrizEnt, Rapido=True))

                ImgRes = Image.fromarray(MatrizRes, "L")

                self.after(0, lambda: self._MostrarResultadoCompleto_MM(
                    ImgIni, ImgRes, MatrizEnt, MatrizRes, Filtro))
            else:
                LblProg = self.LabelProgCompLS
                LblProg.config(text=f"⏳ Aplicando filtro {Filtro} a imagen {W}×{H}…")
                self.update_idletasks()

                MatrizSin = (AplicarFiltroLaplaciano(MatrizEnt, Rapido=True)
                             if Filtro == "laplaciano"
                             else AplicarFiltroSobel(MatrizEnt, Rapido=True))
                MatrizRe  = ReescalarMatrizA255(MatrizSin)
                ImgRes    = Image.fromarray(MatrizRe, "L")

                self.after(0, lambda: self._MostrarResultadoCompleto_LS(
                    ImgIni, ImgRes, MatrizEnt, MatrizSin, MatrizRe, Filtro))
        except Exception as E:
            self.after(0, lambda: messagebox.showerror("Error", str(E)))

    def _MostrarResultadoCompleto_MM(self, ImgIni, ImgRes, MatEnt, MatRes, Filtro):
        self._MostrarImagen(self.LabelCompMM_IniImg, ImgIni, 520)
        self._MostrarImagen(self.LabelCompMM_ResImg, ImgRes, 520)

        H, W = MatRes.shape
        Min, Max, Med = MatRes.min(), MatRes.max(), MatRes.mean()
        Info = (f"Filtro aplicado: {Filtro.upper()}\n"
                f"Tamaño imagen: {W}×{H} píxeles  ({W*H:,} píxeles totales)\n"
                f"Valor mínimo : {Min}\n"
                f"Valor máximo : {Max}\n"
                f"Valor medio  : {Med:.2f}\n\n"
                f"— Primeras 10 filas y 10 columnas de la matriz resultado —\n\n"
                + ConvertirMatrizATexto(MatRes[:10, :10]) +
                f"\n\n(La imagen completa tiene {H} filas × {W} columnas.\n")
        self._EscribirTexto(self.TextoCompMM_Info, Info)
        self.LabelProgCompMM.config(text=f" Filtro {Filtro} aplicado a imagen completa {W}×{H}.")
        self._Estado(f" Filtro '{Filtro}' aplicado a imagen completa ({W}×{H}).", C["Verde"])
        self.NB.select(4)

    def _MostrarResultadoCompleto_LS(self, ImgIni, ImgRes, MatEnt, MatSin, MatRe, Filtro):
        self._MostrarImagen(self.LabelCompLS_IniImg, ImgIni, 520)
        self._MostrarImagen(self.LabelCompLS_ResImg, ImgRes, 520)

        H, W = MatRe.shape
        Min, Max, Med = MatRe.min(), MatRe.max(), MatRe.mean()
        Info = (f"Filtro aplicado: {Filtro.upper()}\n"
                f"Tamaño imagen: {W}×{H} píxeles  ({W*H:,} píxeles totales)\n\n"
                f"── Matriz SIN re-escalar (primeras 10×10) ──\n\n"
                + ConvertirMatrizATexto(MatSin[:10, :10], UsarDecimales=True) +
                f"\n\n── Matriz RE-ESCALADA 0–255 (primeras 10×10) ──\n\n"
                + ConvertirMatrizATexto(MatRe[:10, :10]) +
                f"\n\nValor mínimo re-escalado : {Min}\n"
                f"Valor máximo re-escalado : {Max}\n"
                f"Valor medio  re-escalado : {Med:.2f}\n\n"
                f"(La imagen completa tiene {H} filas × {W} columnas.\n"
                f" Se muestra solo 10×10 por legibilidad.)")
        self._EscribirTexto(self.TextoCompLS_Info, Info)
        self.LabelProgCompLS.config(text=f" Filtro {Filtro} aplicado a imagen completa {W}×{H}.")
        self._Estado(f" Filtro '{Filtro}' aplicado a imagen completa ({W}×{H}).", C["Verde"])
        self.NB.select(5)


# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    App = AplicacionFiltrado()
    App.mainloop()
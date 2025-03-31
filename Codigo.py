import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def sample_and_hold(t_org, signal, fs_ori, fs_dest):
  t_dest = 1/fs_dest
  i = 0
  signal_resample = np.zeros(len(signal))
  samples = []
  t_sample= []
  while i* t_dest < max(t_org):
    idx_sample = np.where((t_org >= i *t_dest) & (t_org < i + 1 * t_dest))[0]
    signal_resample[idx_sample[0]: idx_sample[-1] +1 ] = signal[idx_sample[0]]
    samples.append(signal[idx_sample[0]])
    t_sample.append(i *t_dest)
    i += 1
  return signal_resample, t_sample, samples
def fxquant(s, bit):
    # s : señal de entrada normalizada entre -1 y 1
    # bit : bits de cuantizacion
    Plus1 = np.power(2, (bit - 1))
    X = s * Plus1
    X = np.round(X)
    X = np.minimum(Plus1, X)  # Cambiar el límite superior a Plus1
    X = np.maximum(-1.0 * Plus1, X)
    X = X / Plus1
    return X
def plot_spectrogram(data, fs):
  
  '''
  Parameter:
  data: señal a la que se le calcula el espectrograma
  fs: frecuencia de muestreo de la señal
  '''
  # Resolución del espectrograma
  num =  1024
  hop = num//8

  stft = np.abs(librosa.stft(data,  n_fft=num, hop_length=hop  ))
  D = librosa.amplitude_to_db(stft, ref = np.max)
  spec = librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear', cmap=None, hop_length=hop)

def ejercicio1_1(frecuencia, muestras_por_ciclo, ciclos):
    periodo = 1 / frecuencia  # Periodo de la señal
    t = np.linspace(0, ciclos * periodo, ciclos * muestras_por_ciclo)
    senal = np.sin(2 * np.pi * frecuencia * t)
    return senal, t

def ejercicio1_2():
    f_max = 10_000  # Hz

    # Generamos la señal original (3 períodos) con alta resolución
    muestras_por_ciclo = 1000   # Ajusta este valor para mayor o menor resolución
    ciclos = 3
    senal, t = ejercicio1_1(f_max, muestras_por_ciclo, ciclos)

    # Frecuencia de muestreo "original" (para compatibilidad con la firma de la función)
    fs_ori = f_max * muestras_por_ciclo  

    # Definimos las diferentes frecuencias de muestreo
    fs_list = [f_max/4, f_max/2, f_max, 2*f_max, 4*f_max, 8*f_max, 16*f_max, 32*f_max]

    plt.figure(figsize=(10, 8))

    for i, fs_dest in enumerate(fs_list, start=1):
        # Obtenemos la señal muestreada con sample-and-hold
        s_resample, t_samp, samples = sample_and_hold(t, senal, fs_ori, fs_dest)
        
        plt.subplot(4, 2, i)
        # Señal original
        plt.plot(t, senal, label='Señal original')
        # Señal sample-and-hold (onda escalonada)
        plt.step(t, s_resample, where='post', label='Sample and Hold')
        # Puntos de muestreo (con marcador y tamaño amplificado)
        plt.scatter(t_samp, samples, marker='x', s=80, label='Muestras')
        
        plt.title(f'Muestreo a fs = {fs_dest:.0f} Hz')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.legend()

    plt.tight_layout()
    plt.show()

def ejercicio1_3():
    # Parámetros de la señal
    frecuencia = 1000           # Frecuencia en Hz
    muestras_por_ciclo = 100    # Muestras por cada ciclo
    ciclos = 3                # Número de ciclos de la señal
    
    # Generar la señal original
    senal, t = ejercicio1_1(frecuencia, muestras_por_ciclo, ciclos)
    
    # Lista de bits para cuantización
    bits_list = [1, 2, 3, 4, 5, 6]
    
    # Crear subplots: 3 filas, 2 columnas
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()  # Aplanar para iterar fácilmente
    
    for ax, bits in zip(axes, bits_list):
        # Cuantización de la señal con el número de bits actual
        senal_cuantizada = fxquant(senal, bits)
        
        # Graficar la señal original (línea discontinua) y la cuantizada
        ax.plot(t, senal, label='Señal original', linestyle='--')
        ax.plot(t, senal_cuantizada, label=f'Cuantizada con {bits} bits', marker='o', markersize=3, linewidth=1)
        ax.set_title(f'Cuantización con {bits} bits')
        ax.set_ylabel('Amplitud')
        ax.legend()
        ax.grid(True)
    
    # Eliminar subplots vacíos si hay menos de 6 niveles de cuantización
    for i in range(len(bits_list), len(axes)):
        fig.delaxes(axes[i])
    
    axes[-2].set_xlabel('Tiempo [s]')  # Etiqueta en el penúltimo subplot
    plt.tight_layout()
    plt.show()
def ejercicio2_1(audio_path):
    signal, fs = librosa.load(audio_path, sr=None)
    
    # Convertir a mono si es necesario
    if signal.ndim > 1:
        signal = signal.mean(axis=0)

    # Duración de 40 ms en muestras
    duration_ms = 40
    duration_samples = int((duration_ms / 1000) * fs)

    # Seleccionar 40 ms desde la mitad del audio
    mid_point = len(signal) // 2
    start_sample = max(0, mid_point - duration_samples // 2)
    end_sample = min(len(signal), start_sample + duration_samples)
    signal_segment = signal[start_sample:end_sample]

    fs_dests = [fs/8, fs / 4, fs / 2, fs, fs * 2, fs * 4]

    plt.figure(figsize=(12, 8))

    for i, fs_dest in enumerate(fs_dests, start=1):
        # Muestrear signal_segment con la frecuencia fs_dest
        num_samples = int(len(signal_segment) * fs_dest / fs)
        signal_resample = resample(signal_segment, num_samples)

        # Graficar la señal original y la muestreada
        plt.subplot(3, 2, i)
        t_original = np.linspace(0, duration_ms / 1000, len(signal_segment))
        t_resample = np.linspace(0, duration_ms / 1000, num_samples)
        plt.plot(t_original, signal_segment, label='Original', linestyle='--', color='red', alpha=1)
        plt.plot(t_resample, signal_resample, label=f'Muestreada (fs = {fs_dest:.1f} Hz)', color='blue', linestyle='--', marker='o', markersize=3)
        plt.title(f'Comparación a fs = {fs_dest:.1f} Hz')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

def ejercicio2_2(audio_path):


    data, fs_original = librosa.load(audio_path, sr=None)  # sr=None para conservar Fs original

# Extraemos 500 ms (0.5 segundos):
    segment_duration = 0.5
    N = int(segment_duration * fs_original)
    mid_point = len(data) // 2
    start_sample = max(0, mid_point - N // 2)
    end_sample = min(len(data), start_sample + N)
    segment_500ms = data[start_sample:end_sample]

    # Distintas frecuencias de muestreo a evaluar
    Fs_list = [16000, 8000, 4000, 2000]

    plt.figure(figsize=(10, 8))

    for i, Fs in enumerate(Fs_list):
        # Resampleamos el mismo segmento de 500 ms a Fs
        segment_resampled = librosa.resample(segment_500ms, orig_sr=fs_original, target_sr=Fs)
        
        plt.subplot(len(Fs_list), 1, i+1)
        plot_spectrogram(segment_resampled, Fs)
        
        # Mantener ejes consistentes (hasta 0.5 s en tiempo y ~22 kHz en frecuencia)
        plt.xlim([0, 0.5])    # Hasta 0.5 s
        plt.ylim([0, 4000])  # Hasta aprox la mitad de 44.1 kHz (para comparar en todas las figuras)
        plt.title(f"Espectrograma de 500 ms a Fs = {Fs} Hz")

    plt.tight_layout()
    plt.show()

def ejercicio2_3(audio, fs_original, fs_list):
    
    for fs_target in fs_list:
        # Resamplear la señal completa
        resampled = librosa.resample(audio, orig_sr=fs_original, target_sr=fs_target)

        print(f"\n🔊 Reproduciendo audio a {fs_target} Hz...")
        sd.play(resampled, fs_target)
        sd.wait()  # Esperar a que termine la reproducción

        time.sleep(1)  # Pausa entre audios para claridad
ejercicio2_2("C:/Users/Miguel/sample.wav")

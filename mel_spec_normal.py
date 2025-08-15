import torchaudio
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt

def generate_mel_spectrogram(input_audio_path, output_image_path):
    # 1. Load the audio file
    waveform, sample_rate = torchaudio.load(input_audio_path)
    if waveform.shape[0] > 1: # Check if it's stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)

# 2. Define the Mel-spectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    )

# 3. Apply the transform
    mel_spectrogram = mel_spectrogram_transform(waveform)

# 4. Convert to decibels
    amp_to_db = T.AmplitudeToDB()
    mel_spectrogram_db = amp_to_db(mel_spectrogram) 

# Squeeze the tensor to remove the channel dimension for plotting
    spec_for_plotting = mel_spectrogram_db.squeeze().numpy()
    # 1. Define the output resolution (dots per inch)
#    A larger DPI will result in a larger image file. 100 is a good starting point.
    dpi = 100
# 2. Get the shape of the spectrogram
    height_pixels, width_pixels = spec_for_plotting.shape

# 3. Calculate the dynamic figure size in inches
#    The figure size is the number of pixels divided by the DPI
    fig_width_inches = width_pixels / dpi
    fig_height_inches = 2.5

# 4. Create the plot with the dynamic size
    plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
    plt.imshow(spec_for_plotting, cmap='inferno', aspect='auto', origin='lower')

# Remove axes and padding for a clean image
    plt.axis('off')

# 5. Save the figure
#    The bbox_inches and pad_inches are crucial to remove whitespace
    plt.savefig(output_image_path, pad_inches=0 , bbox_inches='tight', dpi=dpi)

# Close the plot to free up memory
    plt.close()
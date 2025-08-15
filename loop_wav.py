from mel_spec_normal import generate_mel_spectrogram
import os



def main():
    # Define input and output directories
    waves_dir = "/home/baraa/Desktop/testroch/DSED_strong_label/dataset/audio/eval/distorted_lowpass_filter"
    output_dir = "/home/baraa/Desktop/testroch/DSED_strong_label/dataset/audio/eval/distorted_lowpass_filter_mel"

    # Get all wav files from the waves directory
    wav_files = [f for f in os.listdir(waves_dir) if f.endswith('.wav')]
    
    print(f"Found {len(wav_files)} wav files to process...")
    
    # Process each wav file
    for wav_file in wav_files:
        # Create full input path
        input_path = os.path.join(waves_dir, wav_file)
        
        # Create output filename (replace .wav with .png)
        output_filename = wav_file.replace('.wav', '.png')
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {wav_file} -> {output_filename}")
        
        try:
            # Generate mel spectrogram
            generate_mel_spectrogram(input_path, output_path)
            print(f"✓ Successfully generated: {output_filename}")
        except Exception as e:
            print(f"✗ Error processing {wav_file}: {str(e)}")
    
    print("Processing complete!")


if __name__ == "__main__":
    main() 
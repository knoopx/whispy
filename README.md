# Whispy

A voice input tool for Linux that records audio from a microphone, detects voice activity, transcribes speech locally using AI models, and outputs the raw, unprocessed transcription to stdout.

## Features

- **Local AI Transcription**: Uses Distil-Whisper for fast, offline speech-to-text
- **Voice Activity Detection**: Employs silero-vad to automatically stop recording when speech ends
- **PipeWire Integration**: Records audio directly from PipeWire sources
- **Wayland Compatible**: Output can be piped to wtype for seamless integration in Wayland compositors
- **Audio Feedback**: Plays start and stop sounds to indicate recording state
- **Nix Flakes**: Easy installation and dependency management with Nix

## Requirements

- Nix with flakes enabled
- PipeWire audio server
- Wayland compositor (for wtype integration)
- CUDA-compatible GPU (optional, for faster transcription)

## Installation

This project uses Nix flakes for packaging and dependency management.

### Direct Installation

```bash
nix profile add --impure path:.
```

This installs the `whispy` script along with all dependencies.

### Flake Installation

If you manage your system with Nix flakes, you can add whispy to your flake:

```nix
inputs.whispy.url = "github:knoopx/whispy";
inputs.whispy.inputs.nixpkgs.follows = "nixpkgs";

outputs = { whispy, ... }: {
  # Add to your overlays or packages
  nixosConfigurations.yourHost = nixpkgs.lib.nixosSystem {
    modules = [
      {
        nixpkgs.overlays = [
          (self: super: { whispy = whispy.packages.${super.system}.default; })
        ];
        environment.systemPackages = [ pkgs.whispy ];
      }
    ];
  };
};
```

For Home Manager integration:

```nix
home.packages = [ pkgs.whispy ];
```

Then, configure your window manager to bind a key to `whispy`. See the Configuration section for examples.

## Usage

### Basic Usage

Run the whispy script with your audio device target:

```bash
whispy "alsa_input.usb-046d_081b_78B9CE90-02.mono-fallback"
```

Or use the default system input device:

```bash
whispy
```

This will:
1. Play a start sound
2. Record audio from the microphone until 2 seconds of silence
3. Play a stop sound
4. Transcribe the audio and output the text to stdout

### Transcribe Audio File

```bash
whispy audio.wav
```

### Specifying Audio Device

To use a specific PipeWire audio target:

```bash
whispy "alsa_input.usb-Your_Mic-00.mono-fallback"
```

Find available targets with:

```bash
pw-cli list-objects | grep "Audio/Source"
```

### Options

- `--model`: Hugging Face model ID (default: distil-whisper/distil-large-v3.5)
- `--language`: Language code (e.g., 'en', 'es'). Auto-detect if not specified.
- `--timeout`: Silence timeout in seconds (default: 2.0)

## Configuration

### Niri Compositor Integration

To bind a keyboard shortcut in Niri, add to your `~/.config/niri/config.kdl`:

```
binds {
    "Mod+Y".action = {spawn = ["sh", "-c", "whispy alsa_input.usb-046d_081b_78B9CE90-02.mono-fallback | ydotool type -d 0 -f -"];};
}
```

### Audio Device Configuration

The audio target must be specified as an argument. Find your device using:

```bash
pw-cli list-objects | grep "Audio/Source"
```

For example, if your microphone is `alsa_input.usb-046d_081b_78B9CE90-02.mono-fallback`:

```bash
whispy "alsa_input.usb-046d_081b_78B9CE90-02.mono-fallback"
```


## How It Works

1. **Recording**: `pw-record` captures raw audio from PipeWire
2. **Voice Activity Detection**: `pysilero-vad` detects when speech ends
3. **Transcription**: Distil-Whisper converts audio to text
4. **Output**: Prints the transcribed text to stdout (can be piped to `wtype` for typing)

## License

MIT

# 🤟 ASL Hand Gesture Recognition - A to Z

An educational real-time tool that recognizes **American Sign Language (ASL) fingerspelling** from A to Z using your webcam.

---

## 📌 Description

This project uses **MediaPipe HandLandmarker** to detect 21 hand landmarks and classify hand poses into one of **26 ASL letters**. It is designed as an educational tool to help users learn ASL fingerspelling interactively.

---

## ✨ Features

- 🖐️ Real-time hand gesture detection via webcam
- 🔤 Recognizes all 26 ASL alphabet letters (A–Z)
- 📊 Displays finger states (UP/DOWN) live
- 📖 Built-in ASL reference guide panel
- 🔄 Smooth detection (reduces flickering)
- 📱 Responsive UI for different screen sizes
- 🌙 Auto low-light brightness boost
- 🎨 7 background color presets (with tint strength control)
- ⚙️ Settings panel for real-time customization
- 🖌️ Rounded, modern UI with translucent panels

---

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

## 📥 Model Download

Download the required MediaPipe model file and place it in the **same folder** as the script:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

## 🚀 How to Run

```bash
python hand_gesture.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `r` | Toggle ASL reference guide panel |
| `s` | Toggle settings panel |
| `b` | Toggle auto low-light boost |
| `[` | Decrease manual brightness gain |
| `]` | Increase manual brightness gain |
| `-` | Decrease background tint strength |
| `+` | Increase background tint strength |
| `1`–`7` | Select background color preset |
| `q` | Quit the application |

---

## 🎨 Background Color Presets

| Key | Preset | Description |
|-----|--------|-------------|
| `1` | Off | No background tint |
| `2` | Warm Amber | Cozy orange-amber tint |
| `3` | Soft Cyan | Cool studio cyan tint |
| `4` | Mint | Soft mint green tint |
| `5` | Lavender | Purple/lavender tint |
| `6` | Rose | Warm pink tint |
| `7` | Neutral Gray | Neutral brightness lift |

---

## 🌙 Low-Light Enhancement

The program automatically detects dark scenes and boosts brightness. You can also:
- Toggle auto boost with `b`
- Manually adjust gain with `[` and `]`
- Tune the background tint strength with `-` and `+`
- See live scene brightness in the settings panel (`s`)

---

## 🤌 ASL Fingerspelling Guide

| Letter | Hand Sign |
|--------|-----------|
| A | Fist, thumb beside index finger |
| B | Four fingers up, thumb folded |
| C | Curved hand forming "C" shape |
| D | Index up, others touch thumb |
| E | All fingers curled down |
| F | OK sign + 3 fingers up |
| G | Index+thumb pointing sideways |
| H | Index+middle pointing sideways |
| I | Fist with pinky up |
| J | Like I with motion |
| K | Index+middle up spread, thumb between |
| L | L-shape: index up + thumb out |
| M | Fist, thumb under 3 fingers |
| N | Fist, thumb under 2 fingers |
| O | All fingertips touch thumb |
| P | K-sign pointing downward |
| Q | G-sign pointing downward |
| R | Index+middle crossed |
| S | Fist, thumb over fingers |
| T | Thumb between index+middle |
| U | Index+middle up together |
| V | Peace/victory sign |
| W | Index+middle+ring up spread |
| X | Index finger hooked/bent |
| Y | Thumb+pinky out |
| Z | Index finger pointing |

---

## 📁 Project Structure

```
Hand_Gesture/
│
├── hand_gesture.py          # Main script
├── hand_landmarker.task     # MediaPipe model file (download separately)
├── README.md                # English documentation
└── README_BN.md             # Bengali documentation
```

---

## 👥 Contributors

- [ANIKAOSDlang](https://github.com/ANIKAOSDlang)
- [laboni1012](https://github.com/laboni1012)

---

## 📄 License

This project is open source and available for educational use.

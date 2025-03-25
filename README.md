# Video Super-Resolution with Enhancements

## Overview

This project enhances video quality using Super-Resolution techniques with additional video enhancement features, including color correction, noise reduction, sharpness enhancement, and frame stabilization. The model used is ESPCN (Efficient Sub-Pixel Convolutional Neural Network), which upscales video resolution by 4x.

## Features

✅ Super-Resolution: Increases video resolution by 4x using ESPCN.

✅ Color Correction: Enhances brightness and contrast using Histogram Equalization.

✅ Noise Reduction: Removes unwanted noise with Non-Local Means Denoising.

✅ Sharpness Enhancement: Uses Unsharp Masking for clearer visuals.

✅ Frame Stabilization: Reduces jitter using Optical Flow-based stabilization.

## Technologies Used

**1. Python**

**2. OpenCV (for computer vision processing)**

**3. Deep Learning (ESPCN model)**

**4. NumPy (for efficient matrix operations)**

## Installation

1. Clone the repository:

```git clone https://github.com/yourusername/video-super-resolution.git
cd video-super-resolution
```
2. Install dependencies:
```
pip install opencv-python numpy
```
3. Download the ESPCN_x4.pb model file and place it in the project directory.

## Usage

Run the Video Super-Resolution Model

```
python super_resolution.py --input input_video.mp4 --output high_res_video.mp4
```
--input : Path to the input video.

--output : Path to save the enhanced video.

## Code Overview

* Super-Resolution Processing
```py
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x4.pb")
sr.setModel("espcn", 4)  # Using ESPCN model with 4x upscaling
```
* Video Enhancement Techniques
```py
def apply_color_correction(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```
* Frame Stabilization
```py
def stabilize_frame(prev_frame, current_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.remap(current_frame, flow[:, :, 1], flow[:, :, 0], cv2.INTER_LINEAR)
```
## Results

The processed videos demonstrate significant improvements in resolution, clarity, and smoothness, making them ideal for enhanced viewing experiences in low-resolution video scenarios.

## Future Improvements

🔹 Implement Real-ESRGAN for even better results.
🔹 Add support for real-time video enhancement.
🔹 Optimize GPU acceleration for faster processing.

## Contributing

Feel free to fork the repo, suggest improvements, or submit pull requests! 🚀

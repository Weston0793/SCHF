# Pediatric Supracondylar Humerus X-Ray Fracture Detector

This application is designed for research and educational purposes, using AI models to detect fractures in pediatric supracondylar humerus X-rays. It employs a Twin Convolutional Neural Network to enhance and analyze X-ray images.

## Disclaimer

**This application is for research and educational purposes only.**
**The AI models utilized herein may produce inaccurate or unreliable results.**
**Always consult a medical professional for clinical diagnosis and treatment.**

## Features

- Upload X-ray images in JPG, PNG, or JPEG formats.
- Enhance uploaded images using adaptive histogram equalization, sharpening, and contrast stretching.
- Automatically crop the region of interest in the X-ray image.
- Generate predictions for fractures with confidence scores.
- Visualize Class Activation Maps (CAM) to highlight areas of interest in the X-ray.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Weston0793/SCHF.git
    cd SCHF
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained models and place them in the `models` folder. (Refer to the repository or contact the authors for the models.)

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run webapp.py
    ```

2. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

3. Use the interface to upload an X-ray image, and view the enhanced image, cropped image, predictions, and Class Activation Map (CAM).

Alternatively, you can check out the hosted version of the application at [SCHF Diagnostics](https://schfdiagnostics.streamlit.app/).

## Authors

- Aba Lőrincz<sup class='superscript'>1,2,3,*</sup>
- András Kedves<sup class='superscript'>2</sup>
- Hermann Nudelman<sup class='superscript'>1,3</sup>
- András Garami<sup class='superscript'>1</sup>
- Gergő Józsa<sup class='superscript'>1,3</sup>
- Zsolt Kisander<sup class='superscript'>2</sup>

## Affiliations

1. Department of Thermophysiology, Institute for Translational Medicine, Medical School, University of Pécs, 12 Szigeti Street, H7624 Pécs, Hungary; aba.lorincz@gmail.com (AL)
2. Department of Automation, Faculty of Engineering and Information Technology, University of Pécs, 2 Boszorkány Street, H7624 Pécs, Hungary
3. Division of Surgery, Traumatology, Urology, and Otorhinolaryngology, Department of Paediatrics, Clinical Complex, University of Pécs, 7 József Attila Street, H7623 Pécs, Hungary

## Code

The source code for this project is available on GitHub: [GitHub Repository](https://github.com/Weston0793/SCHF/)

## License

This project is licensed under the GNU License. See the [LICENSE](LICENSE) file for more details.

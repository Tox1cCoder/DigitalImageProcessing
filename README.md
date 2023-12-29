# Digital Image Processing [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

A web application built with Streamlit for processing and enhancing low-quality images. The application consists of two modules:

## Image Restoration:
- Supports restoration of images with three degradations: low light, noisy, and blurry.

## Image Enhancement:
- Utilizes Real ESRGAN and BSRGAN for enhancing images.

## Installation:
1. Clone this repository and navigate to the project directory.
2. Run the command `pip install -r requirements.txt` to install the dependencies.

## Usage:
1. Run the command `streamlit run app.py` to start the application.
2. Open your web browser and go to http://localhost:8501.
3. By default, Streamlit allows file uploads up to **200MB**. If you want to increase the maximum upload size for audio files, use the command `streamlit run app.py --server.maxUploadSize=1028`.
4. Use the application to upload and process low-light images.

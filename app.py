import os

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from app_funcs import *


def lowLight():
    upload_path = "uploads/"
    download_path = "downloads/"
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())

        with st.spinner(f"Enhancing... 💫"):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
            enhance_image(uploaded_image, downloaded_image)

            final_image = Image.open(downloaded_image)
            print("Opening ", final_imageaa)

            image_comparison(
                img1=uploaded_image,
                img2=final_image,
                label1="Original",
                label2="Enhanced",
            )
            with open(downloaded_image, "rb") as file:
                if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                    if st.download_button(
                            label="Download Enhanced Image",
                            data=file,
                            file_name=str("enhanced_" + uploaded_file.name),
                            mime='image/jpg'
                    ):
                        download_success()

                if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                    if st.download_button(
                            label="Download Enhanced Image",
                            data=file,
                            file_name=str("enhanced_" + uploaded_file.name),
                            mime='image/jpeg'
                    ):
                        download_success()

                if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                    if st.download_button(
                            label="Download Enhanced Image",
                            data=file,
                            file_name=str("enhanced_" + uploaded_file.name),
                            mime='image/png'
                    ):
                        download_success()

                if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                    if st.download_button(
                            label="Download Enhanced Image",
                            data=file,
                            file_name=str("enhanced_" + uploaded_file.name),
                            mime='image/bmp'
                    ):
                        download_success()
    else:
        st.warning('⚠ Please upload your Image file')


def diffBIR():
    pass


def main():
    st.set_page_config(
        page_title="Image Quality Enhancer",
        page_icon="✨",
        layout="centered",
        initial_sidebar_state="auto",
    )
    main_image = Image.open('static/main_banner.png')

    st.image(main_image, use_column_width='auto')
    st.title("Image Quality Enhancer ")
    st.info('Supports all popular image formats - PNG, JPG, BMP')

    st.title("Select Activity")
    choice = st.selectbox("Menu", (
        "About", "Low Light Processor", "Quality"))

    if choice == "Low Light Processor":
        lowLight()

    elif choice == "Quality":
        diffBIR()

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()

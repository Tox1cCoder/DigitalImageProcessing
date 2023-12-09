import os
from streamlit_image_comparison import image_comparison
from app_funcs import *


def Image_Restoration(method):
    upload_path = "uploads/"
    download_path = "downloads/"
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])
    
    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())

        with st.spinner(f"Enhancing... "):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
            
            if method == "Low Light":
                enhance_image(uploaded_image, downloaded_image)
            elif method == "Blurry":
                NAFNetBlur(uploaded_image)
            elif method == "Denoise":
                NAFNetNoise(uploaded_image)

            final_image = Image.open(downloaded_image)
            print("Opening ", final_image)

            image_comparison(
                img1=uploaded_image,
                img2=final_image,
                label1="Original",
                label2="Enhanced",
            )
            file_extension = uploaded_file.name.split(".")[-1]
            with open(downloaded_image, "rb") as file:
                if st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name=str("enhanced_" + uploaded_file.name),
                        mime=f'image/{file_extension}'
                ):
                    download_success()

    else:
        st.warning('⚠ Please upload your Image file')


def Super_Resolution(method):
    upload_path = "uploads/"
    download_path = "downloads/"
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())

        with st.spinner(f"Enhancing... "):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
            if method == "BSRGan":
                # supper_resolution(model_path="models\BSRGAN.pth", uploaded_image=uploaded_image)
                supper_resolution(model_path="/content/drive/MyDrive/models/BSRGANx2.pth", uploaded_image=uploaded_image)
            elif method == "RealESRGAN+":
                # sr_real_esrgan(model_path="models\RealESRGAN_x2plus.pth", input_path=uploaded_image, scale=2)
                sr_real_esrgan(model_path="/content/drive/MyDrive/models/RealESRGAN_x4plus.pth", input_path=uploaded_image, scale=4)
            
            print(downloaded_image)
            final_image = Image.open(downloaded_image)
            print("Opening ", final_image)

            image_comparison(
                img1=uploaded_image,
                img2=final_image,
                label1="Original",
                label2="Enhanced",
            )
            file_extension = uploaded_file.name.split(".")[-1]
            with open(downloaded_image, "rb") as file:
                if st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name=str("enhanced_" + uploaded_file.name),
                        mime=f'image/{file_extension}'
                ):
                    download_success()

    else:
        st.warning('⚠ Please upload your Image file')


def main():
    st.set_page_config(
        page_title="Image Quality Enhancer",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("Image Quality Enhancer ")

    st.title("Configuration")
    choice = st.selectbox("Menu", (
        "About", "Image Restoration", "Quality Enhancment"))

    if choice == "Image Restoration":
        method = st.selectbox("Method", ("Low Light", "Blurry", "Denoise"))
        Image_Restoration(method=method)

    elif choice == "Quality Enhancment":
        method = st.selectbox("Method", ("BSRGan", "RealESRGAN+"))
        Super_Resolution(method=method)

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()

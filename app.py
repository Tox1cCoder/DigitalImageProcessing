import os
from streamlit_image_comparison import image_comparison
from app_funcs import *
import torch
import gc

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
                torch.cuda.empty_cache()
            elif method == "Blurry":
                NAFNetBlur(uploaded_image)
                torch.cuda.empty_cache()
            elif method == "Denoise":
                NAFNetNoise(uploaded_image)
                torch.cuda.empty_cache()

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
            
        uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
        downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
        
        if method == "BSRGan":
            types = st.selectbox("Types:", ("Scale x1", "Scale x2"))
            if st.button("Process"):
                with st.spinner(f"Enhancing... "):
                    super_resolution(uploaded_image=uploaded_image, types=types)
                    torch.cuda.empty_cache()

        elif method=="RealESRGAN+":
            types = st.selectbox("Types:", ("Scale x2", "Scale x4", "Anime x4"))
            face =st.checkbox("Face enhance")
            if st.button("Process"):
                with st.spinner(f"Enhancing... "):
                    sr_real_esrgan(input_path=uploaded_image, scale=4, types=types, face_enhance=face)
                    torch.cuda.empty_cache()

            # print(downloaded_image)
            # final_image = Image.open(downloaded_image)
            # print("Opening ", final_image)

        image_comparison(
            img1=uploaded_image,
            img2=downloaded_image,
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
        "About", "Image Restoration", "Quality Enhancement"))

    if choice == "Image Restoration":
        method = st.selectbox("Method", ("Low Light", "Blurry", "Denoise"))
        Image_Restoration(method=method)

    elif choice == "Quality Enhancement":
        method = st.selectbox("Method", ("BSRGan", "RealESRGAN+"))
        Super_Resolution(method=method)

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()

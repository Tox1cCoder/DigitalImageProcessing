import os
from streamlit_image_comparison import image_comparison
from app_funcs import *


def lowLight():
    upload_path = "uploads/"
    download_path = "downloads/"
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])
    
    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())

        with st.spinner(f"Enhancing... "):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
            enhance_image(uploaded_image, downloaded_image)

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


def Super_Resolution():
    upload_path = "uploads/"
    download_path = "downloads/"
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "bmp", "jpeg"])
    
    if uploaded_file is not None:
        with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
            f.write((uploaded_file).getbuffer())

        with st.spinner(f"Enhancing... "):
            uploaded_image = os.path.abspath(os.path.join(upload_path, uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path, str("enhanced_" + uploaded_file.name)))
            supper_resolution(uploaded_image)

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
    st.info('Supports all popular image formats - PNG, JPG, BMP')

    st.title("Select Activity")
    choice = st.selectbox("Menu", (
        "About", "Low Light Processor", "Quality"))

    if choice == "Low Light Processor":
        lowLight()

    elif choice == "Quality":
        Super_Resolution()

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()

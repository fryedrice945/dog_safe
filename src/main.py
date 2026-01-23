import io

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dog Safe Rekognition", layout="centered")


def _show_image_preview(image_bytes, filename="photo"):
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, caption=filename, width="stretch")
    width, height = img.size
    st.write(f"Dimensions: {width} x {height}px")
    st.download_button(
        label="Download image",
        data=image_bytes,
        file_name=filename,
        mime="image/png",
    )


def run_app():
    st.title("Dog Safe Rekognition")
    st.write("Upload a photo or take one with your device camera. Mobile friendly.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader(
            "Upload a photo",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
    with col2:
        camera_img = st.camera_input("Take a photo")

    image_bytes = None
    filename = None

    if uploaded is not None:
        image_bytes = uploaded.read()
        filename = uploaded.name
    elif camera_img is not None:
        image_bytes = camera_img.getvalue()
        filename = "camera_photo.png"

    if image_bytes and filename:
        _show_image_preview(image_bytes, filename)

    st.markdown("---")
    st.write("Privacy: images are processed locally in your browser/session.")


def lambda_handler(event, context):
    run_app()
    return {"statusCode": 200, "body": "hello from lambda"}


if __name__ == "__main__":
    lambda_handler(None, None)
# ...existing code...

import io
import json
import os

import boto3
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dog Safe", layout="centered")


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


def send_to_bedrock(image_bytes: bytes):
    try:
        aws_cfg = st.secrets.get("aws", {})
        session = boto3.Session(
            aws_access_key_id=aws_cfg.get("access_key_id"),
            aws_secret_access_key=aws_cfg.get("secret_access_key"),
            region_name=aws_cfg.get("region_name", "us-east-1"),
        )
        client = session.client(service_name="bedrock-runtime", region_name="us-east-1")

        system_prompt = [
            {
                "text": (
                    "You are an image analysis model that can analyze images and extract the text "
                    "from within the image. You are also an expert in canine dietary safety. "
                    "Given an image of an ingredient label, you will extract the text from the "
                    "label and determine if the food is safe for dogs to consume based on common "
                    "dietary guidelines. Also base your answer on scientific research and veterinary "
                    "recommendations."
                ),
            },
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                ],
            },
        ]
        request_body = {
            "system": system_prompt,
            "messages": messages,
        }

        return client.invoke_model(
            modelId=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0"),
            body=json.dumps(request_body),
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")


def run_app():
    st.title("Dog Safe 🐶🛡️")
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
        with st.spinner("Analyzing image..."):
            response = send_to_bedrock(image_bytes)
        if response:
            response_body = json.loads(response["body"].read())
            st.markdown("### Analysis Result")
            for message in response_body.get("messages", []):
                if message.get("role") == "assistant":
                    for content in message.get("content", []):
                        if "text" in content:
                            st.write(content["text"])

    st.markdown("---")
    st.write("Privacy: images are processed locally in your browser/session.")


def lambda_handler(event, context):
    run_app()
    return {"statusCode": 200, "body": "hello from lambda"}


if __name__ == "__main__":
    lambda_handler(None, None)

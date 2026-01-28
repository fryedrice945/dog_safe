import base64
import io
import json
import os
import re

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


def send_to_bedrock(image_bytes: bytes, mime_type: str):
    try:
        aws_cfg = st.secrets.get("aws", {})
        session = boto3.Session(
            aws_access_key_id=aws_cfg.get("ACCESS_KEY"),
            aws_secret_access_key=aws_cfg.get("ACCESS_SECRET"),
            region_name=aws_cfg.get("region_name", "us-east-1"),
        )
        client = session.client(service_name="bedrock-runtime", region_name="us-east-1")

        system_prompt = [
            {
                "text": (
                    "You are an image-analysis assistant. Perform OCR on the supplied image (an ingredient label) and parse the ingredient declaration. "
                    "Respond ONLY with a single valid JSON object and nothing else, UTF-8 encoded, with these keys: "
                    '"determination" — one of "safe", "unsafe", or "uncertain"; '
                    '"confidence" — one of "low", "medium", or "high"; '
                    '"harmful_ingredients" — array of any explicitly listed ingredients known to be toxic to dogs (lowercase); '
                    '"extracted_ingredients" — array of parsed ingredient tokens in listed order; '
                    '"extracted_text" — full OCR text from the label; '
                    '"explanation" — one concise sentence (<=20 words) justifying the determination without repeating the harmful_ingredients list; '
                    '"recommended_action" — brief action when determination is "unsafe" or "uncertain" (e.g., "consult a veterinarian"); '
                    "Rules: base judgments ONLY on ingredients explicitly listed; do NOT hallucinate or infer unlisted ingredients; "
                    'if the label is ambiguous, incomplete, or unreadable set determination="uncertain" and confidence="low" with recommended_action="consult a veterinarian"; '
                    'Set confidence to "high" when the label clearly lists ingredients and contains or clearly omits known toxins, "medium" for possible omissions, "low" for unclear/poor OCR. '
                    "Keep the JSON minimal and machine-parseable."
                ),
            },
        ]
        image_format = mime_type.split("/")[-1] if mime_type else "png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": {"format": image_format, "source": {"bytes": image_b64}}},
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
    st.write(
        "Upload a photo of an ingredient label or take one with your device camera to determine if it's safe for dogs. Mobile friendly.",
    )

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
    mime_type = None

    if uploaded is not None:
        image_bytes = uploaded.read()
        filename = uploaded.name
        mime_type = getattr(uploaded, "type", "image/png")
    elif camera_img is not None:
        image_bytes = camera_img.getvalue()
        filename = "camera_photo.png"
        mime_type = getattr(camera_img, "type", "image/png")

    if image_bytes and filename:
        _show_image_preview(image_bytes, filename)
        with st.spinner("Analyzing image..."):
            response = send_to_bedrock(image_bytes, mime_type)
        if response:
            body_bytes = response["body"].read()
            content_type = response.get("contentType", "")

            if "application/json" in content_type:
                parsed = json.loads(body_bytes.decode("utf-8"))

                parsed = parsed.get("output").get("message").get("content")[0]
            elif content_type.startswith("text/") or content_type == "":
                parsed = body_bytes.decode("utf-8")
            else:
                parsed = body_bytes

            # normalize to text for display
            if isinstance(parsed, dict):
                result_text = (
                    parsed.get("text") or parsed.get("content") or json.dumps(parsed)
                )
            else:
                result_text = parsed

            # try to pull out a Determination line
            # ensure result_text is a str for regex operations
            if isinstance(result_text, (bytes, bytearray)):
                result_text = result_text.decode("utf-8", errors="replace")
            else:
                result_text = str(result_text)

            det_match = re.search(
                r"Determination:\s*(.+)",
                result_text,
                flags=re.IGNORECASE,
            )
            if det_match:
                explanation = re.sub(
                    r"Determination:\s*.+",
                    "",
                    result_text,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                explanation = result_text

            st.subheader("Analysis")
            st.markdown(explanation)

    st.markdown("---")
    st.write("Privacy: images are processed locally in your browser/session.")


def lambda_handler(event, context):
    run_app()
    return {"statusCode": 200, "body": "hello from lambda"}


if __name__ == "__main__":
    lambda_handler(None, None)

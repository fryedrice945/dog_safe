import io
import json
import os
import base64
import boto3
import re
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
                    "You are an image analysis model that can analyze images and extract the text "
                    "from within the image. You are also an expert in canine dietary safety. "
                    "Given an image of an ingredient label, you will extract the text from the "
                    "label and determine if the food is safe for dogs to consume based on common "
                    "dietary guidelines. Also base your answer on scientific research and veterinary "
                    "recommendations. If the label contains any ingredients that are known to be harmful "
                    "to dogs, such as chocolate, grapes, raisins, onions, garlic, or artificial "
                    "sweeteners like xylitol, you should flag the food as unsafe. If the label is "
                    "not clear or if you cannot determine the safety of the food based on the provided "
                    "information, respond with 'uncertain' and recommend consulting a veterinarian. "
                    "Provide a concise explanation for your determination."
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
            determination = None
            if det_match:
                determination = det_match.group(1).splitlines()[0].strip()
                # remove the determination line from the main explanation for cleaner display
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

            if determination:
                dlow = determination.lower()
                if "unsafe" in dlow or "not safe" in dlow or "danger" in dlow:
                    st.error(f"Determination: {determination}")
                elif "uncertain" in dlow or "unknown" in dlow:
                    st.warning(f"Determination: {determination}")
                elif "safe" in dlow or "ok" in dlow:
                    st.success(f"Determination: {determination}")
                else:
                    st.info(f"Determination: {determination}")

            with st.expander("Raw model output"):
                if isinstance(parsed, (dict, list)):
                    st.code(json.dumps(parsed, indent=2), language="json")
                else:
                    st.code(result_text)

    st.markdown("---")
    st.write("Privacy: images are processed locally in your browser/session.")


def lambda_handler(event, context):
    run_app()
    return {"statusCode": 200, "body": "hello from lambda"}


if __name__ == "__main__":
    lambda_handler(None, None)

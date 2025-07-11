import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image
from lenscribe.core import VlitImageProcessor
import io


def read_image(upload: UploadedFile) -> Image.Image:
    image_bytes = upload.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def main():
    st.set_page_config(page_title="Lenscribe - Vision & Language API", layout="wide")
    st.title("Lenscribe - Vision & Language API")


    st.write("Upload an image to ask questions about it.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = read_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        question = st.text_input("Ask a question about the image:")
        if st.button("Get Answer"):
            if question:
                image_processor =  VlitImageProcessor(question=question, image_path=uploaded_file.name)
                image_processor.configure()
                inputs = image_processor.processor(image, question, return_tensors="pt")
                outputs = image_processor.model.generate(**inputs)
                answer = image_processor.processor.decode(outputs[0], skip_special_tokens=True)
                st.write(f"**Answer:** {answer}")
            else:
                st.error("Please enter a question.")
    else:
        st.info("Please upload an image to start.")
        st.stop()


if __name__ == "__main__":
    main()


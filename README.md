# lenscribe
Lenscribe is an intelligent vision-language application powered by Python that bridges the gap between images and natural language. Leveraging state-of-the-art deep learning models, Lenscribe can understand, describe, and extract insights from visual content in real-time. Whether it's captioning images, answering questions about scenes, or generating contextual text based on visual inputs, Lenscribe empowers users with seamless multimodal interaction.

## Quickstart

Run the Lenscribe Streamlit app to interact with images and ask questions:

```bash
streamlit run main.py
```

1. Upload an image (`.jpg`, `.jpeg`, or `.png`).
2. Enter a question about the image (e.g., "What is happening in this picture?").
3. Click **Get Answer** to receive an AI-generated response.

Make sure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

The app uses the `VlitImageProcessor` from `lenscribe.core` to process images and answer questions using state-of-the-art vision-language models.

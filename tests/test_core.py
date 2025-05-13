from lenscribe.core import BlipImageProcessor

def test_blip_image_processor():
    """
    Test the BlipImageProcessor class.
    """
    question = "What is going on in this picture?"
    image_path = "tests/test_picture_2.jpg"
    processor = BlipImageProcessor(question, image_path)
    
    # Test configuration
    processor.configure()
    assert processor.processor is not None, "Processor should be configured."
    assert processor.model is not None, "Model should be configured."
    
    # Test processing
    answer = processor.process()
    assert answer is not None, "The answer should not be None."
    assert isinstance(answer, str), "The answer should be a string."
    assert len(answer) > 0, "The answer should not be empty."
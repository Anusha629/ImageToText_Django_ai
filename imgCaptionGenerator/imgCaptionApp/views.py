from django.shortcuts import render
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load the pre-trained models and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Modify the predict_step function to use processor
def predict_step(image_paths, num_captions=1):
    captions = []  # Initialize captions as an empty list

    # Process uploaded images and generate captions
    try:
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        # Process images with the processor
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate captions
        output_ids = model.generate(pixel_values, **gen_kwargs, num_return_sequences=num_captions)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        captions = preds  # Assign the generated captions
    except Exception as e:
        # Handle exceptions or errors here
        print(f"Error: {str(e)}")

    return captions

def caption_view(request):
    captions = []  # Initialize captions as an empty list

    if request.method == 'POST':
        uploaded_images = request.FILES.getlist('images')
        if uploaded_images:
            captions = predict_step([image.file for image in uploaded_images])
    
    return render(request, 'index.html', context={'captions': captions})



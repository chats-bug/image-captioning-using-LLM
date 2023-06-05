import gradio as gr
import torch
from PIL import Image

from image_descriptor_models import BlipBaseModel, GitBaseCocoModel, Blip2Model, OpenClipModel
from language_models import OpenAIModels, OpenAIModelNames

IMAGE_TEXT_MODELS = {
	"Git-Base-COCO": GitBaseCocoModel,
	"Blip Base": BlipBaseModel,
	"Blip 2": Blip2Model,
	"OpenClip": OpenClipModel,
}

TEXT_MODELS = {
	"OpenAI": OpenAIModels,
	"Raven": None,
}

# examples = [["Image1.png"], ["Image2.png"], ["Image3.png"]]

def generate_captions(
	image,
	num_captions,
	model_name,
	max_length,
	temperature,
	top_k,
	top_p,
	repetition_penalty,
	diversity_penalty,
	):
	"""
	Generates captions for the given image.

	-----
	Parameters:
	image: PIL.Image
		The image to generate captions for.
	num_captions: int
		The number of captions to generate.
	** Rest of the parameters are the same as in the model.generate method. **
	-----
	Returns:
	list[str]
	"""
	# Convert the numerical values to their corresponding types.
	# Gradio Slider returns values as floats: except when the value is a whole number, in which case it returns an int.
	# Only float values suffer from this issue.
	temperature = float(temperature)
	top_p = float(top_p)
	repetition_penalty = float(repetition_penalty)
	diversity_penalty = float(diversity_penalty)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = IMAGE_TEXT_MODELS[model_name](device)

	captions = model.generate(
		image=image,
		max_length=max_length,
		num_captions=num_captions,
		temperature=temperature,
		top_k=top_k,
		top_p=top_p,
		repetition_penalty=repetition_penalty,
		diversity_penalty=diversity_penalty,
	)

	language_model = OpenAIModels(
		model=OpenAIModelNames.GPT_3_5_TURBO,
	)
	instagram_ready_captions = language_model.generate_alternate_captions(
		descriptions=captions,
		num_captions=num_captions,
		length=max_length,
	)

	return instagram_ready_captions

title = "AI tool for generating captions for images"
description = "This tool uses pretrained models to generate captions for images."

interface = gr.Interface(
	fn=generate_captions,
	inputs=[
		gr.components.Image(type="pil", label="Image"),
		gr.components.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Captions to Generate"),
		gr.components.Dropdown(IMAGE_TEXT_MODELS.keys(), label="Model", value=list(IMAGE_TEXT_MODELS.keys())[1]), # Default to Blip Base
		gr.components.Slider(minimum=20, maximum=100, step=5, value=50, label="Maximum Caption Length"),
		gr.components.Slider(minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Temperature"),
		gr.components.Slider(minimum=1, maximum=100, step=1, value=50, label="Top K"),
		gr.components.Slider(minimum=0.1, maximum=5.0, step=0.1, value=1.0, label="Top P"),
		gr.components.Slider(minimum=1.0, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"),
		gr.components.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="Diversity Penalty"),
	],
	outputs=[
		gr.components.Textbox(label="Caption"),
	],
	# Set image examples to be displayed in the interface.
	examples = [
		["Image1.png", 1, list(IMAGE_TEXT_MODELS.keys())[1], 50, 1.0, 50, 1.0, 2.0, 2.0],
		["Image2.png", 1, list(IMAGE_TEXT_MODELS.keys())[1], 50, 1.0, 50, 1.0, 2.0, 2.0],
		["Image3.png", 1, list(IMAGE_TEXT_MODELS.keys())[1], 50, 1.0, 50, 1.0, 2.0, 2.0],
	],
	title=title,
	description=description,
	allow_flagging="never",
)


if __name__ == "__main__":
    # Launch the interface.
	interface.launch(
		enable_queue=True,
		debug=True,
	)
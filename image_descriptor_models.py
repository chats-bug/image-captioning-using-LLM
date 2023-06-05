from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration, VisionEncoderDecoderModel
import torch
import open_clip


class ImageCaptionModel:
	def __init__(
		self,
		device,
		processor,
		model,
	) -> None:
		"""
		Initializes the model for generating captions for images.

		-----
		Parameters:
		device: str
			The device to use for the model. Must be either "cpu" or "cuda".
		processor: transformers.AutoProcessor
			The preprocessor to use for the model.
		model: transformers.AutoModelForCausalLM or transformers.BlipForConditionalGeneration
			The model to use for generating captions.

		-----
		Returns:
		None
		"""
		self.device = device
		self.processor = processor
		self.model = model
		self.model.to(self.device)

	def generate(
		self,
		image,
		num_captions: int = 1,
		max_length: int = 50,
		temperature: float = 1.0,
		top_k: int = 50,
		top_p: float = 1.0,
		repetition_penalty: float = 1.0,
		diversity_penalty: float = 0.0,
	):
		"""
		Generates captions for the given image.

		-----
		Parameters:
		preprocessor: transformers.PreTrainedTokenizerFast
			The preprocessor to use for the model.
		model: transformers.PreTrainedModel	
			The model to use for generating captions.
		image: PIL.Image
			The image to generate captions for.
		num_captions: int
			The number of captions to generate.
		temperature: float
			The temperature to use for sampling. The value used to module the next token probabilities that will be used by default in the generate method of the model. Must be strictly positive. Defaults to 1.0.
		top_k: int
			The number of highest probability vocabulary tokens to keep for top-k-filtering. A large value of top_k will keep more probabilities for each token leading to a better but slower generation. Defaults to 50.
		top_p: float
			The value that will be used by default in the generate method of the model for top_p. If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
		repetition_penalty: float
			The parameter for repetition penalty. 1.0 means no penalty. Defaults to 1.0.
		diversity_penalty: float
			The parameter for diversity penalty. 0.0 means no penalty. Defaults to 0.0.

		"""
		# Type checking and making sure the values are valid.
		assert type(num_captions) == int and num_captions > 0, "num_captions must be a positive integer."
		assert type(max_length) == int and max_length > 0, "max_length must be a positive integer."
		assert type(temperature) == float and temperature > 0.0, "temperature must be a positive float."
		assert type(top_k) == int and top_k > 0, "top_k must be a positive integer."
		assert type(top_p) == float and top_p > 0.0, "top_p must be a positive float."
		assert type(repetition_penalty) == float and repetition_penalty >= 1.0, "repetition_penalty must be a positive float greater than or equal to 1."
		assert type(diversity_penalty) == float and diversity_penalty >= 0.0, "diversity_penalty must be a non negative float."

		pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device) # Convert the image to pixel values.

		# Generate captions ids.
		if num_captions == 1:
			generated_ids = self.model.generate(
				pixel_values=pixel_values,
				max_length=max_length,
				num_return_sequences=1,
				temperature=temperature,
				top_k=top_k,
				top_p=top_p,
			)
		else:
			generated_ids = self.model.generate(
				pixel_values=pixel_values,
				max_length=max_length,
				num_beams=num_captions, # num_beams must be greater than or equal to num_captions and must be divisible by num_beam_groups.
				num_beam_groups=num_captions, # num_beam_groups is set to equal to num_captions so that all the captions are diverse
				num_return_sequences=num_captions, # generate multiple captions which are very similar to each other due to the grouping effect of beam search.
				temperature=temperature,
				top_k=top_k,
				top_p=top_p,
				repetition_penalty=repetition_penalty,
				diversity_penalty=diversity_penalty,
			)
			
		# Decode the generated ids to get the captions.
		generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

		return generated_caption


class GitBaseCocoModel(ImageCaptionModel):
	def __init__(self, device):
		"""
		A wrapper class for the Git-Base-COCO model. It is a pretrained model for image captioning.

		-----
		Parameters:
		device: str
			The device to run the model on, either "cpu" or "cuda".
		checkpoint: str
			The checkpoint to load the model from.

		-----
		Returns:
		None
		"""
		checkpoint = "microsoft/git-base-coco"
		processor = AutoProcessor.from_pretrained(checkpoint)
		model = AutoModelForCausalLM.from_pretrained(checkpoint)
		super().__init__(device, processor, model)


class BlipBaseModel(ImageCaptionModel):
	def __init__(self, device):
		"""
		A wrapper class for the Blip-Base model. It is a pretrained model for image captioning.

		-----
		Parameters:
		device: str
			The device to run the model on, either "cpu" or "cuda".
		checkpoint: str
			The checkpoint to load the model from.

		-----
		Returns:
		None
		"""
		self.checkpoint = "Salesforce/blip-image-captioning-base"
		processor = AutoProcessor.from_pretrained(self.checkpoint)
		model = BlipForConditionalGeneration.from_pretrained(self.checkpoint)
		super().__init__(device, processor, model)


class Blip2Model(ImageCaptionModel):
	def __init__(self, device):
		"""
		A wrapper class for the OpenClip model. It is a pretrained model for image captioning.

		-----
		Parameters:
		device: str
			The device to run the model on, either "cpu" or "cuda".
		checkpoint: str
			The checkpoint to load the model from.

		-----
		Returns:
		None
		"""
		self.checkpoint = "openai/clip-vit-base-patch32"
		processor = AutoProcessor.from_pretrained(self.checkpoint)
		model = AutoModelForCausalLM.from_pretrained(self.checkpoint)
		processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
		model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b", device_map="auto", load_in_8bit=True)
		super().__init__(device, processor, model)

	
class OpenClipModel():
	def __init__(self, device):
		"""
		A wrapper class for the OpenClip model. It is a pretrained model for image captioning.

		-----
		Parameters:
		device: str
			The device to run the model on, either "cpu" or "cuda".
		checkpoint: str
			The checkpoint to load the model from.

		-----
		Returns:
		None
		"""
		coca_model, _, coca_transform = open_clip.create_model_and_transforms(
			model_name="coca_ViT-L-14",
			pretrained="mscoco_finetuned_laion2B-s13B-b90k"
		)
		self.model = coca_model
		self.transform = coca_transform
		self.device = device

	def generate(self, image, **kwargs):
		"""
		Generates captions for the given image.

		-----
		Parameters:
		image: PIL.Image
			The image to generate captions for.
		kwargs: dict
			Keyword arguments to pass to the model.
		"""
		im = self.transform(image).unsqueeze(0).to(self.device)
		with torch.no_grad(), torch.cuda.amp.autocast():
			generated = self.model.generate(im, seq_len=20)
		return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")

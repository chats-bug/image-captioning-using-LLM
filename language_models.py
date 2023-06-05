import openai
import os
import enum

openai_api_key = "OPENAI_API_KEY"

class OpenAIModelNames(enum.Enum):
	GPT_3_5_TURBO = "gpt-3.5-turbo"
	GPT_3_5 = "gpt-3.5"
	GPT_3 = "gpt-3"
	DAVINCI = "davinci"


class OpenAIModels():
	def __init__(self, prompt: str = None, model: OpenAIModelNames = None) -> None:
		if prompt is None:
			self.prompt = "I will give you some image descriptions generated using AI from an image. Write <n> alternate instagram worthy captions. The max length of each caption should be <length> words."
		else:
			self.prompt = prompt
		if model is None:
			self.model = OpenAIModelNames.GPT_3_5_TURBO
		else:
			self.model = model
		pass

	def generate_alternate_captions(
		self, 
		descriptions: list[str],
		num_captions: int = 5,
		length: int = 30,
	):
		# Replace the <n> and <length> in the prompt with the actual values.
		prompt = self.prompt.replace("<n>", str(num_captions)).replace("<length>", str(length))
		# Add the descriptions to the prompt.
		descriptions_str = "\n".join(descriptions)
		prompt = f"{prompt}\nThe image descriptions are: {descriptions_str}"

		# Generate the captions
		completion = openai.ChatCompletion.create(
			model=self.model,
			messages=[
				{
					"role": "user", 
					"content": prompt
				}
			]
		)

		return completion.choices[0].message


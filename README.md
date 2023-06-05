# AI Image captioning tool
*A tool used to generate social media like captions for images. Check out the demo [here](https://vimeo.com/manage/videos/832600239)*

## Introduction

This report discusses an approach for generating Instagram-worthy captions by combining image descriptions and large language models (LLMs). The objective is to produce captivating and stylistic captions for images, taking into consideration the limited resources available. The report outlines the approach using existing resources and presents an ideal scenario if additional resources were provided.


## Approach with Limited Resources

Given the limited resources, fine-tuning small models with a limited dataset may not yield satisfactory results. Instead, the proposed approach involves utilizing existing models, such as GIT base and BLIP base, to generate image descriptions. These models, while lacking the desired style, can accurately describe the content of the images.

To enhance the generated descriptions and infuse them with a more engaging and stylistic tone, a large language model like GPT-3 or similar models can be leveraged. The generated image descriptions serve as prompts to the language model, requesting it to produce Instagram-worthy captions based on the provided descriptions. This approach takes advantage of the creativity and language generation capabilities of the language model.


## **Benefits and Considerations**

1. **Improved Caption Quality:** By incorporating a powerful language model, the quality of the generated captions can be significantly enhanced, resulting in more engaging and stylistic content suitable for Instagram.
2. **Resource Efficiency:** Utilizing existing models and leveraging the capabilities of a language model allows for resource efficiency, as it minimizes the need for extensive fine-tuning with large datasets.
3. **Prompt Refinement:** Experimenting with different prompts, adjusting parameters, and refining the generated captions can help tailor the results to align with the desired style and content.


## Conclusion

The proposed approach of using image descriptions generated by existing models and large language models in tandem can overcome the limitations of limited resources to produce Instagram-worthy captions. By leveraging the creative abilities of the language models, the quality and stylistic appeal of the generated captions can be significantly improved. With additional resources, such as a curated dataset and fine-tuning, the approach can be further enhanced, aligning the generated captions more closely with the desired style and tone. Continuous evaluation and refinement based on human feedback are essential to ensure the captions meet the expectations of the target audience.
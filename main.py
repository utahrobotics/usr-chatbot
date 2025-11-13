from dotenv import load_dotenv

load_dotenv(".env")

import discord
import os, sys
import json
from typing import Optional
from pydantic import BaseModel


class LLMResponseImageGen(BaseModel):
    message: str
    image_prompt: str | None = None


class LLMResponseNoImageGen(BaseModel):
    message: str


from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

disable_sd = "DISABLE_SD" in os.environ

print(LLMResponseImageGen.model_json_schema())
structured_outputs = StructuredOutputsParams(json=LLMResponseNoImageGen.model_json_schema() if disable_sd else LLMResponseImageGen.model_json_schema())
llm = LLM(model="cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit", gpu_memory_utilization=0.7, max_model_len=15000)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=1000, structured_outputs=structured_outputs)

if not disable_sd:
    import torch
    from diffusers import StableDiffusion3Pipeline
    import random

    # torch.set_float32_matmul_precision("high")

    # torch._inductor.config.conv_1x1_as_mm = True
    # torch._inductor.config.coordinate_descent_tuning = True
    # torch._inductor.config.epilogue_fusion = False
    # torch._inductor.config.coordinate_descent_check_all_directions = True

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16
    ).to("cuda")
    # pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float16)

    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    # pipe.transformer.to(memory_format=torch.channels_last)
    # pipe.vae.to(memory_format=torch.channels_last)

    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
    # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

    # # Warm Up
    # print("Warming up")
    # for _ in range(3):
    #     _ = pipe(prompt="a photo of a cat holding a sign that says hello world", generator=torch.manual_seed(1))
else:
    pipe = None

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')


@client.event
async def on_message(message: discord.message.Message):
    if message.author == client.user:
        return
    
    if client.user not in message.mentions:
        return

    messages = []
    async for message in message.channel.history(limit=20):
        if message.author == client.user:
            messages.append({"role": "assistant", "content": message.content })

        else:
            content = [
                {
                    "type": "text",
                    "text": json.dumps({ "username": message.author.display_name, "message": message.clean_content })
                }
            ]

            for attachment in message.attachments:
                if attachment.filename.endswith(".png") or \
                        attachment.filename.endswith(".webp") or \
                        attachment.filename.endswith(".jpg") or \
                        attachment.filename.endswith(".jpeg"):

                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": attachment.proxy_url
                        }
                    })
            
            messages.append({"role": "user", "content": content})
    
    background = f'''# Background
Your name is T.E.R.I, a funny AI Chatbot for the Utah Student Robotics discord server, named after the existing robot T.E.R.I.
The user messages come in JSON with fields "username" and "message", and the messages are in channel name: {message.channel.name}. You do not have access
to other channels.'''
    
    if pipe is None:
        system_message = f'''{background}

# Style Guide
You only output valid JSON in the following format:
{{
    "message": <a playful and brief message to send>
}}
You do not have the ability to generate images right now. That feature is under maintenance.
'''
    else:
        system_message = f'''{background}

# Style Guide
You only output valid JSON in the following format:
{{
    "message": <a playful and brief message to send>
}}

# Image Generation
If the user *explicitly* requests an image to be generated, write the image generation prompt
in the "image_prompt" field in your output JSON. This is an optional field. You are not allowed to create sexual or harmful images since
this is a public server. The image will be attached to the message.

If the prompt that the user is asking for is unsafe, do not generate the image and let the user
know what was wrong.

You do not have the ability to look at the images you previously generated.
'''

    messages.append(
        {
            "role": "system",
            "content": system_message
        }
    )
    messages.reverse()
    
    async with message.channel.typing():
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={ "enable_thinking": False },
            use_tqdm=False
        )

        response = json.loads(outputs[0].outputs[0].text)

        if "image_prompt" in response:
            assert not disable_sd
            image_prompt = response["image_prompt"]
            print(f'Generating image: "{image_prompt}"')
            image = pipe(prompt=image_prompt, generator=torch.manual_seed(random.randint(0, 10240))).images[0]
            image.save("/tmp/tmp_sd_output.png")
            await message.channel.send(response["message"], file=discord.File("/tmp/tmp_sd_output.png"))
        else:
            await message.channel.send(response["message"])
    

token = os.getenv('DISCORD_TOKEN')
if not token:
    sys.exit("Environment variable DISCORD_TOKEN not set")

client.run(token)

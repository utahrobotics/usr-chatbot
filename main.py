from dotenv import load_dotenv

load_dotenv(".env")

import discord
import os, sys
import json

from vllm import LLM, SamplingParams

llm = LLM(model="cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit", gpu_memory_utilization=0.7, max_model_len = 15000)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=1000)

import torch
from diffusers import StableDiffusion3Pipeline
import random

torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
).to("cuda")


pipe.set_progress_bar_config(disable=True)

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# Warm Up
for _ in range(3):
    _ = pipe(prompt="a photo of a cat holding a sign that says hello world", generator=torch.manual_seed(1))

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

    messages = [
    ]
    async for message in message.channel.history(limit=20):
        if message.author == client.user:
            content = [
                {
                    "type": "text",
                    "text": message.content
                }
            ]

            for attachment in message.attachments:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": attachment.proxy_url
                    }
                })
            
            messages.append({"role": "assistant", "content": content })

        else:
            content = [
                {
                    "type": "text",
                    "text": json.dumps({ "username": message.author.display_name, "message": message.clean_content })
                }
            ]

            for attachment in message.attachments:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": attachment.proxy_url
                    }
                })
            
            messages.append({"role": "user", "content": content})
    
    messages.append(
        {
            "role": "system",
            "content": f'''# Background
Your name is T.E.R.I, a funny AI Chatbot for the Utah Student Robotics discord server, named after the existing robot T.E.R.I.
The user messages come in JSON with fields "username" and "message", and the messages are in channel name: {message.channel.name}. You do not have access
to other channels.

# Style Guide
You only output valid JSON in the following format:
{{
    "message": <a playful and brief message to send>,
    "image_prompt": <OPTIONAL. A prompt to create an image with>
}}

# Image Generation
If the user *explicitly* requests an image to be generated, write the image generation prompt
in the "image_prompt" field in your output JSON. You are not allowed to create sexual or harmful images since
this is a public server. The image will be attached to the message.

If the prompt that the user is asking for is unsafe, do not generate the image and let the user
know what was wrong
'''
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

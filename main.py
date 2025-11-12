import discord
import os, sys
from dotenv import load_dotenv
import json

from vllm import LLM, SamplingParams

# llm = LLM(model="Qwen/Qwen3-8B-AWQ", gpu_memory_utilization=0.8)
# llm = LLM(model="Qwen3VL-8B-Instruct-Q8_0/Qwen3VL-8B-Instruct-Q8_0.gguf", gpu_memory_utilization=0.8)
llm = LLM(model="cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit", gpu_memory_utilization=0.6, max_model_len = 24000)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=600)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

load_dotenv(".env")


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
            messages.append({"role": "assistant", "content": message.content})
        else:
            messages.append({"role": "user", "content": json.dumps({ "username": message.author.display_name, "message": message.clean_content })})
    
    messages.append(
        {
            "role": "system",
            "content": f'''Your name is T.E.R.I, a funny AI Chatbot for the Utah Student Robotics discord server, named after the existing robot T.E.R.I.
The user messages come in JSON with fields "username" and "message", and the messages are in channel name: {message.channel.name}. You do not have access
to other channels. Respond playfully and briefly.'''
        }
    )
    messages.reverse()
    # print(messages)
    
    async with message.channel.typing():
        outputs = llm.chat(
            messages,
            sampling_params=sampling_params,
            chat_template_kwargs={ "enable_thinking": False },
            use_tqdm=False
        )
        
    await message.channel.send(outputs[0].outputs[0].text)


token = os.getenv('DISCORD_TOKEN')
if not token:
    sys.exit("Environment variable DISCORD_TOKEN not set")

client.run(token)

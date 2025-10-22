from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen3-8B-AWQ", gpu_memory_utilization=0.8)

guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=None, temperature=0.7, top_p=0.8, top_k=20, min_p=0)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params
)
print(outputs[0].outputs[0].text)

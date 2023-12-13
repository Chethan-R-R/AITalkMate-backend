from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from transformers import AutoTokenizer, pipeline, logging, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import emoji

class ChatGenerator:
    def __init__(self, model_name_or_path, model_basename, use_triton=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.streamer = TextIteratorStreamer(self.tokenizer)
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None
        )
        logging.set_verbosity(logging.CRITICAL)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            streamer=self.streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )

    def generate(self, user_message, model_reply, prompt):
        system_message = "I know you are AI model"
        prompt_template = f'''<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {model_reply}</s><s>[INST] {prompt} [/INST]'''
        generation_kwargs = dict(text_inputs=prompt_template)
        thread = Thread(target=self.pipe, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        temp = ""
        i = 0
        with ThreadPoolExecutor() as executor:
            for new_text in self.streamer:
                new_text = new_text.strip()
                i += 1
                if i < 3:
                    continue
                if new_text == "" or new_text[-1] == "*" or new_text[0] == "*":
                    continue
                if new_text[-1] == '>':
                    new_text = new_text.rstrip('</s>')
                generated_text += new_text + " "
                new_text = emoji.replace_emoji(new_text, replace='')
                temp += " " + new_text
                if temp[-1] == '.' or temp[-1] == '?' or temp[-1] == '!':
                    executor.submit(tts, temp)
                    temp = ""

        print(generated_text)
        model_reply = generated_text
        user_message = prompt

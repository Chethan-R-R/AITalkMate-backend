from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from transformers import AutoTokenizer, pipeline, logging, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM
import emoji
from tts_api import StyleTTS
from lip_api import Wav2LipInference
import audio

styletts_obj = StyleTTS()
lipSync = Wav2LipInference(checkpoint_path = "checkpoints/wav2lip_gan.pth")
model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
streamer = TextIteratorStreamer(tokenizer)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

logging.set_verbosity(logging.CRITICAL)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15)

def generate(self, user_message, model_reply, prompt):
    system_message = "I know you are AI model"
    prompt_template = f'''<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST] {model_reply}</s><s>[INST] {prompt} [/INST]'''
    generation_kwargs = dict(text_inputs=prompt_template)
    thread = Thread(target=self.pipe, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    temp = ""
    i = 0
    file_id = 0
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
                executor.submit(tts, temp, str(file_id))
                file_id+=1
                temp = ""
    print(generated_text)

def tts(text,file_id):
    wav = styletts_obj.inference(text, alpha=0.0, beta=1.111, diffusion_steps=15, embedding_scale=0.99)
    audio.save_wav(wav, 'temp/'+file_id+'.wav', 24000)
    lipSync.inference(face="avatarframe.jpg",file_id=file_id)
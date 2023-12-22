from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
import emoji
import audio
from tts_api import StyleTTS
from lip_api import Wav2LipInference

class ChatGenerator:
    def __init__(self,model_name_or_path="TheBloke/Llama-2-7b-Chat-GPTQ", model_basename="model", use_triton=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.streamer = TextIteratorStreamer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                        device_map="auto",
                        trust_remote_code=False,
                        revision="main")
        self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                streamer=self.streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1
            )
        self.StyleTTS = StyleTTS()
        self.LipSync = Wav2LipInference()

    def prompt_encode(self, messages):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>", "</s>"
        DEFAULT_SYSTEM_PROMPT = f"""I know you are AI model but while talking to me talk like a sweet friendly english mate susan. I am here to learn english converation make me talk more. please don't generate long responses, only short and sweet responses. Involve me in coversation with you. Figure out by your self what to talk and ask questions."""

        messages = [ B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + messages[0] ] + messages[1:]
        messages_list = [
            f"{BOS}{B_INST} {(prompt).strip()} {E_INST} {(answer).strip()} {EOS}"
            for prompt, answer in zip(messages[::2], messages[1::2])
        ]
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]).strip()} {E_INST}")

        return "".join(messages_list)

    def tts(self,text, directory_name, file_id,client_socket):
        wav = self.StyleTTS.inference(text, alpha=0.523, beta=1.123, diffusion_steps=4, embedding_scale=1)
        audio.save_wav(wav, directory_name+file_id+'.wav', 24000)
        self.LipSync.inference(directory_name=directory_name, file_id=file_id,client_socket=client_socket)

    def generate(self, prompt_list, client_socket, directory_name):
        system_message = "I know you are AI model but while talking to me talk like a sweet friendly english mate susan. I am here to learn english converation make me talk more. please don't generate long responses, only short and sweet responses. Involve me in coversation with you. Figure out by your self what to talk and ask questions"
        prompt_template = self.prompt_encode(prompt_list)
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
                    executor.submit(self.tts, temp, directory_name, str(file_id), client_socket)
                    file_id+=1
                    temp = ""
            if temp!="" and temp!=" ":
                executor.submit(self.tts, temp+'.', directory_name, str(file_id), client_socket)
                file_id+=1
                temp = ""
        print(generated_text)
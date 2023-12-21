from os import path
import numpy as np
import cv2
import subprocess
import torch
from tqdm import tqdm
import audio
from lip_models import Wav2Lip
import platform

class Wav2LipInference:
    def __init__(self, checkpoint_path="checkpoints/wav2lip_gan.pth",face = "avatar.png", fps=25.0,
                 face_det_batch_size=16, wav2lip_batch_size=128, resize_factor=1):
        self.args = {
            'checkpoint_path': checkpoint_path,
            'face':cv2.imread(face),
            'fps': fps,
            'face_det_batch_size': face_det_batch_size,
            'wav2lip_batch_size': wav2lip_batch_size,
            'resize_factor': resize_factor,
            'img_size': 96,
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))
        self.model = self.load_model(checkpoint_path)
        self.frame_h, self.frame_w = self.args['face'].shape[:-1]
        self.frame_half = self.frame_h//2

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def datagen(self, frames, mels):
        img_batch, mel_batch = [], []

        face = frames[0]

        for i, m in enumerate(mels):
            face = cv2.resize(face, (self.args['img_size'], self.args['img_size']))
            img_batch.append(face)
            mel_batch.append(m)

            if len(img_batch) >= self.args['wav2lip_batch_size']:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args['img_size'] // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch,
                                      [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch
                img_batch, mel_batch = [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args['img_size'] // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch

    def sharpen_image(self,image, kernel_size=(5, 5), sigma=0.6, amount=1.5, threshold=0):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        return sharpened

    async def send_msg(self,directory_name,file_id,client_socket):
        with open(directory_name+file_id+'.mp4', "rb") as f:
            file_contents = f.read()
        await client_socket.send(file_contents)

    def inference(self,directory_name, file_id,client_socket):
        full_frames = [self.args['face']]
        fps = self.args['fps']

        wav = audio.load_wav(directory_name+file_id+'.wav', 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + 16])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.args['wav2lip_batch_size']
        gen = self.datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                print("Model loaded")

                out = cv2.VideoWriter(directory_name+file_id+'.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (self.frame_w,self.frame_half), isColor=True)

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p in pred:
                p = self.sharpen_image(p)
                p = cv2.resize(p.astype(np.uint8),(self.frame_w,self.frame_h))
                p = p[self.frame_h-self.frame_half:,]
                out.write(p)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 0 {}'.format(directory_name+file_id+'.wav', directory_name+file_id+'.avi', directory_name+file_id+'.mp4')

        subprocess.call(command, shell=platform.system() != 'Windows')
        self.send_msg(directory_name,file_id,client_socket)

from os import path
import numpy as np
import cv2
import os
import subprocess
import torch
from tqdm import tqdm
import audio
import face_detection
from lip_models import Wav2Lip
import platform

class Wav2LipInference:
    def __init__(self, checkpoint_path="checkpoints/wav2lip_gan.pth", static=False, fps=30.0,
                 face_det_batch_size=16, wav2lip_batch_size=128, resize_factor=2, crop=[0, -1, 0, -1],
                 box=[-1, -1, -1, -1], rotate=False, nosmooth=False):
        self.args = {
            'checkpoint_path': checkpoint_path,
            'static': static,
            'fps': fps,
            'face_det_batch_size': face_det_batch_size,
            'wav2lip_batch_size': wav2lip_batch_size,
            'resize_factor': resize_factor,
            'crop': crop,
            'box': box,
            'rotate': rotate,
            'nosmooth': nosmooth,
            'img_size': 96,
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))

        self.model = self.load_model(checkpoint_path)
        self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=self.device)

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

    def face_detect(self, images):
        batch_size = self.args['face_det_batch_size']

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(self.detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            y1 = rect[1]
            y2 = rect[3]
            x1 = rect[0]
            x2 = rect[2]
            h = (y2 - y1)//12
            w = (x2-x1)//15
            y1 = y1+h
            y2 = y2+h//2
            x1 = x1-w
            x2 = x2+w
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.args['nosmooth']:
            boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        # del self.detector
        return results

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.args['box'][0] == -1:
            if not self.args['static']:
                face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.args['box']
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.args['static'] else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (self.args['img_size'], self.args['img_size']))
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args['wav2lip_batch_size']:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args['img_size'] // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch,
                                      [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args['img_size'] // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def sharpen_image(self,image, kernel_size=(5, 5), sigma=0.6, amount=1.5, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        # sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def inference(self,face,file_id):
        self.args['face'] = face
        # self.args['audio'] = audio_file
        if not os.path.isfile(self.args['face']):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif self.args['face'].split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(self.args['face'])]
            fps = self.args['fps']

        else:
            video_stream = cv2.VideoCapture(self.args['face'])
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.args['resize_factor'] > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // self.args['resize_factor'],
                                               frame.shape[0] // self.args['resize_factor']))

                if self.args['rotate']:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.args['crop']
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        print("Number of frames available for inference: " + str(len(full_frames)))

        # if not self.args['audio'].endswith('.wav'):
        #     print('Extracting raw audio...')
        #     command = 'ffmpeg -y -i "{}" -acodec pcm_s16le -ar 16000 -ac 1 -strict -2 {}'.format(self.args['audio'], 'temp/temp.wav')
        #     subprocess.call(command, shell=True)
        #     self.args['audio'] = 'temp/temp.wav'

        wav = audio.load_wav('temp/'+file_id+'.wav', 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

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

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(np.ceil(
                                                                            float(len(mel_chunks)) / batch_size)))):
            if i == 0:
                print("Model loaded")

                frame_h, frame_w = full_frames[0].shape[:-1]
                y1, y2, x1, x2 = coords[0]
                half = (y2-y1)//2
                out = cv2.VideoWriter('temp/'+file_id+'.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (x2-x1,half), isColor=True)


            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                p = self.sharpen_image(p)
                p = cv2.resize(p.astype(np.uint8),(x2-x1,y2-y1))
                p = p[(y2-y1)-half:,]
                out.write(p)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 0 {}'.format('temp/'+file_id+'.wav', 'temp/'+file_id+'.avi','results/'+file_id+'.mp4')

        subprocess.call(command, shell=platform.system() != 'Windows')



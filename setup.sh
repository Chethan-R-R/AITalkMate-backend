#!/bin/bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
pip install -q youtube-dl

# Clone StyleTTS2-LibriTTS repository with Git LFS
git lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

# Move the contents of the Models directory to the project root
mv StyleTTS2-LibriTTS/Models .

# Download wav2lip_gan.pth
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O '/content/AITalkMateBackend/checkpoints/wav2lip_gan.pth'

# Install ghc package
pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl

pip install ffmpeg-python
pip install numpy
pip install opencv-contrib-python
pip install opencv-python
pip install torchvision
pip install numba
pip install librosa==0.9.1
#!/bin/bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
# Clone StyleTTS2-LibriTTS repository with Git LFS
git lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS

# Move the contents of the Models directory to the project root
mv StyleTTS2-LibriTTS/Models .

# Download wav2lip_gan.pth
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O '/content/AITalkMateBackend/checkpoints/wav2lip_gan.pth'

# Install ghc package
pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl

# Download s3fd.pth for face detection
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/content/AITalkMateBackend/face_detection/detection/sfd/s3fd.pth"

sudo apt-get update && sudo apt-get install -y software-properties-common ffmpeg
sudo apt-get update && apt-get install -y ffmpeg

sudo add-apt-repository universe
sudo apt-get install qt5-default qttools5-dev -y
sudo apt-get install python3-tk -y
pip3 install git+https://github.com/openai/whisper.git && echo
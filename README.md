# Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation
This repo contains the official PyTorch implementation of  [*Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation*](https://pages.cs.huji.ac.il/adiyoss-lab/TempoTokens/)

![alt text](https://github.com/guyyariv/TempoTokens/blob/master/audio-to-video.mp4)

# Abstract
We consider the task of generating diverse and realistic videos guided by natural audio samples from
a wide variety of semantic classes. For this task, the videos are required to be aligned both
globally and temporally with the input audio: globally, the input audio is semantically associated
with the entire output video, and temporally, each segment of the input audio is associated with a
corresponding segment of that video. We utilize an existing text-conditioned video generation model
and a pre-trained audio encoder model. The proposed method is based on a lightweight adaptor network,
which learns to map the audio-based representation to the input representation expected by the
text-to-video generation model. As such, it also enables video generation conditioned on text, audio,
and, for the first time as far as we can ascertain, on both text and audio.
We validate our method extensively on three datasets demonstrating significant semantic diversity
of audio-video samples and further propose a novel evaluation metric (AV-Align) to assess
the alignment of generated videos with input audio samples. AV-Align is based on the detection and
comparison of energy peaks in both modalities. In comparison to recent state-of-the-art approaches,
our method generates videos that are better aligned with the input sound, both with respect to
content and temporal axis. We also show that videos produced by our method present higher visual
quality and are more diverse.

<a href="https://arxiv.org/abs/XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXXX-b31b1b.svg" height=22.5></a>
<a href="https://pages.cs.huji.ac.il/adiyoss-lab/TempoTokens/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

[//]: # ([![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/GuyYariv/AudioToken&#41;)

# Installation
```
git clone git@github.com:guyyariv/TempoTokens.git
cd TempoTokens
pip install -r requirements.txt
```
And initialize an Accelerate environment with:
```angular2html
accelerate config
```
Download BEATs pre-trained model 
```
mkdir -p models/BEATs/ && wget "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" -P "models/BEATs/"
```

# Training
Execute the relevant command for each dataset we have trained on, including [VGGSound](https://huggingface.co/datasets/Loie/VGGSound/tree/main), [Landscape](https://drive.google.com/drive/folders/14A1zaQI5EfShlv3QirgCGeNFzZBzQ3lq), and [AudioSet-Drum](https://www.dropbox.com/s/7ykgybrc8nb3lgf/AudioSet_Drums.zip?dl=0).
```angular2html
accelerate launch train.py --config configs/v2/vggsound.yaml
```
```angular2html
accelerate launch train.py --config configs/v2/landscape.yaml
```
```angular2html
accelerate launch train.py --config configs/v2/audioset_drum.yaml
```
We strongly recommend reviewing the configuration files and customizing the parameters according to your preferences.

# Inference

The ```inference.py``` script serves the purpose of generating videos using trained checkpoints.
Once you've completed the model training using the provided command (or opted for our pre-trained models)
, you can effortlessly create videos from the datasets we've utilized for training, such as
[VGGSound](https://huggingface.co/datasets/Loie/VGGSound/tree/main), 
[Landscape](https://drive.google.com/drive/folders/14A1zaQI5EfShlv3QirgCGeNFzZBzQ3lq), 
and [AudioSet-Drum](https://www.dropbox.com/s/7ykgybrc8nb3lgf/AudioSet_Drums.zip?dl=0).
```angular2html
accelerate launch inference.py --mapper_weights models/TempoTokens/learned_embeds.pth --testset vggsound
```
```angular2html
accelerate launch inference.py --mapper_weights models/landscape/learned_embeds.pth --testset landscape
```
```angular2html
accelerate launch inference.py --mapper_weights models/audioset_drum/learned_embeds.pth --testset audioset_drum
```
Moreover, you have the capability to generate a video from your own audio, as demonstrated below:
```angular2html
accelerate launch inference.py --mapper_weights models/TempoTokens/learned_embeds.pth --audio_path /audio/path
```

# Acknowledgments
Our code is partially built upon [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning)

# Cite
If you use our work in your research, please cite the following paper:
```
```

# License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file. 


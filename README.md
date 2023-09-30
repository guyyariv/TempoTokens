# Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation
This repo contains the official PyTorch implementation of  [*Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation*](https://pages.cs.huji.ac.il/adiyoss-lab/TempoTokens/)

https://github.com/guyyariv/TempoTokens/assets/89798559/753cc371-33a6-4574-b049-0f570f07a389


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

<a href="https://arxiv.org/abs/2309.16429"><img src="https://img.shields.io/badge/arXiv-2309.16429-b31b1b.svg" height=22.5></a>
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
Download [BEATs](https://github.com/microsoft/unilm/blob/master/beats/BEATs.py) pre-trained model 
```
mkdir -p models/BEATs/ && wget -P models/BEATs/ -O "models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
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

# Pre-trained weights
Obtain the pre-trained weights for the three datasets we conducted training on by visiting the following link: https://drive.google.com/drive/folders/10pRWoq0m5torvMXILmIQd7j9fLPEeHtS
We advise you to save the folders in the directory named "models/."

# Inference

The ```inference.py``` script serves the purpose of generating videos using trained checkpoints.
Once you've completed the model training using the provided command (or opted for our pre-trained models)
, you can effortlessly create videos from the datasets we've utilized for training, such as
[VGGSound](https://huggingface.co/datasets/Loie/VGGSound/tree/main), 
[Landscape](https://drive.google.com/drive/folders/14A1zaQI5EfShlv3QirgCGeNFzZBzQ3lq), 
and [AudioSet-Drum](https://www.dropbox.com/s/7ykgybrc8nb3lgf/AudioSet_Drums.zip?dl=0).
```angular2html
accelerate launch inference.py --mapper_weights models/vggsound/learned_embeds.pth --testset vggsound
```
```angular2html
accelerate launch inference.py --mapper_weights models/landscape/learned_embeds.pth --testset landscape
```
```angular2html
accelerate launch inference.py --mapper_weights models/audioset_drum/learned_embeds.pth --testset audioset_drum
```
Moreover, you have the capability to generate a video from your own audio, as demonstrated below:
```angular2html
accelerate launch inference.py --mapper_weights models/vggsound/learned_embeds.pth --audio_path /audio/path
```

```
> python inference.py --help

usage: inference.py [-h] -m MODEL -p PROMPT [-n NEGATIVE_PROMPT] [-o OUTPUT_DIR]
                    [-B BATCH_SIZE] [-W WIDTH] [-H HEIGHT] [-T NUM_FRAMES]
                    [-WS WINDOW_SIZE] [-VB VAE_BATCH_SIZE] [-s NUM_STEPS]
                    [-g GUIDANCE_SCALE] [-i INIT_VIDEO] [-iw INIT_WEIGHT] [-f FPS]
                    [-d DEVICE] [-x] [-S] [-lP LORA_PATH] [-lR LORA_RANK] [-rw]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        HuggingFace repository or path to model checkpoint directory
  -p PROMPT, --prompt PROMPT
                        Text prompt to condition on
  -n NEGATIVE_PROMPT, --negative-prompt NEGATIVE_PROMPT
                        Text prompt to condition against
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save output video to
  -B BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for inference
  -W WIDTH, --width WIDTH
                        Width of output video
  -H HEIGHT, --height HEIGHT
                        Height of output video
  -T NUM_FRAMES, --num-frames NUM_FRAMES
                        Total number of frames to generate
  -WS WINDOW_SIZE, --window-size WINDOW_SIZE
                        Number of frames to process at once (defaults to full
                        sequence). When less than num_frames, a round robin diffusion
                        process is used to denoise the full sequence iteratively one
                        window at a time. Must be divide num_frames exactly!
  -VB VAE_BATCH_SIZE, --vae-batch-size VAE_BATCH_SIZE
                        Batch size for VAE encoding/decoding to/from latents (higher
                        values = faster inference, but more memory usage).
  -s NUM_STEPS, --num-steps NUM_STEPS
                        Number of diffusion steps to run per frame.
  -g GUIDANCE_SCALE, --guidance-scale GUIDANCE_SCALE
                        Scale for guidance loss (higher values = more guidance, but
                        possibly more artifacts).
  -i INIT_VIDEO, --init-video INIT_VIDEO
                        Path to video to initialize diffusion from (will be resized to
                        the specified num_frames, height, and width).
  -iw INIT_WEIGHT, --init-weight INIT_WEIGHT
                        Strength of visual effect of init_video on the output (lower
                        values adhere more closely to the text prompt, but have a less
                        recognizable init_video).
  -f FPS, --fps FPS     FPS of output video
  -d DEVICE, --device DEVICE
                        Device to run inference on (defaults to cuda).
  -x, --xformers        Use XFormers attnetion, a memory-efficient attention
                        implementation (requires `pip install xformers`).
  -S, --sdp             Use SDP attention, PyTorch's built-in memory-efficient
                        attention implementation.
  -lP LORA_PATH, --lora_path LORA_PATH
                        Path to Low Rank Adaptation checkpoint file (defaults to empty
                        string, which uses no LoRA).
  -lR LORA_RANK, --lora_rank LORA_RANK
                        Size of the LoRA checkpoint's projection matrix (defaults to
                        64).
  -rw, --remove-watermark
                        Post-process the videos with LAMA to inpaint ModelScope's
                        common watermarks.
```

# Acknowledgments
Our code is partially built upon [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning)

# Cite
If you use our work in your research, please cite the following paper:
```
@misc{yariv2023diverse,
      title={Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation}, 
      author={Guy Yariv and Itai Gat and Sagie Benaim and Lior Wolf and Idan Schwartz and Yossi Adi},
      year={2023},
      eprint={2309.16429},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file. 


# MIT License

# Copyright (c) 2023 Hans Brouwer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import os
import warnings
from typing import List, Optional
import torchaudio
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip

import numpy as np
from PIL import Image
import torch
from compel import Compel
from diffusers import DPMSolverMultistepScheduler
from modules.pipelines.pipeline_audio_to_video import TextToVideoSDPipeline
from modules.unet.unet_3d_condition import UNet3DConditionModel

from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange

from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark
from utils.lora import inject_inferable_lora


def combine_video_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Set the audio of the video clip to the provided audio
    video = video.set_audio(audio)

    output_path = video_path[:-4] + '_c.mp4'

    try:
        # Write the final video with audio to the output path
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except:
        pass


def cut_wav_file(input_path, output_path, start_time, end_time):
    # Load the WAV file
    audio, sr = torchaudio.load(input_path)

    # Calculate the start and end samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the desired portion of the audio
    extracted_audio = audio[:, start_sample:end_sample]

    # Save the extracted audio as a new WAV file
    torchaudio.save(output_path, extracted_audio, sr)


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet, beats, at_embedder = load_primary_models(model)
        # Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens("<temp>")
        num_added_tokens += tokenizer.add_tokens("<local1>")
        num_added_tokens += tokenizer.add_tokens("<local2>")
        num_added_tokens += tokenizer.add_tokens("<local3>")
        num_added_tokens += tokenizer.add_tokens("<local4>")
        num_added_tokens += tokenizer.add_tokens("<class>")
        if num_added_tokens < 6:
            raise ValueError(
                f"The tokenizer already contains the token <*>. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        del _unet  # This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet")
        at_embedder.eval()
        at_embedder.load_state_dict(torch.load(args.embedder_weights))
        beats.eval()

        y, sr = torchaudio.load(output_audio)

        desired_sample_rate = 16000
        if sr != desired_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sample_rate)
            y = resampler(y)
            sr = desired_sample_rate

        # Step 2: Convert the stereo audio to mono
        if len(y) == 2:
            y = torch.mean(y, dim=0, keepdim=True)

        args.num_frames = int(y.shape[1] / sr) * args.fps
        audio = y

        with torch.no_grad():
            audio_features = beats.extract_features(audio)[1]
            temporal_token, local_window_1, local_window_2, local_window_3, \
            local_window_4, audio_token = at_embedder(audio_features)

    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet.disable_gradient_checkpointing()
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()

    inject_inferable_lora(pipe, lora_path, r=lora_rank, unet_replace_modules=['UNet3DConditionModel'])

    return pipe, temporal_token, [local_window_1, local_window_2, local_window_3, local_window_4], audio_token


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):
    if init_video is None:
        # initialize with random gaussian noise
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape, dtype=torch.half)

    else:
        # encode init_video to latents
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in trange(
        0, latents.shape[0], batch_size, desc="Decoding to pixels...", unit_scale=batch_size, unit="frame"
    ):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()


def primes_up_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    latents: Tensor,
    init_weight: float,
    prompt: Optional[List[str]],
    negative_prompt: Optional[List[str]],
    prompt_embeds: Optional[List[Tensor]],
    negative_prompt_embeds: Optional[List[Tensor]],
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
    rotate: bool,
    temporal_token=None,
    audio_token=None,
    local_windows=None
):
    device = pipe.device
    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size, _, num_frames, _, _ = latents.shape
    window_size = min(num_frames, window_size)

    prompt_embeds = pipe._encode_prompt(
        audio_token=audio_token,
        temporal_token=temporal_token,
        local_windows=local_windows,
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=negative_prompt_embeds,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # set the scheduler to start at the correct timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_step = round(init_weight * len(pipe.scheduler.timesteps))
    timesteps = pipe.scheduler.timesteps[start_step:]
    if init_weight == 0:
        latents = torch.randn_like(latents)
    else:
        latents = pipe.scheduler.add_noise(
            original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
        )

    # manually track previous outputs for the scheduler as we continually change the section of video being diffused
    model_outputs = [None] * order

    if rotate:
        shifts = np.random.permutation(primes_up_to(window_size))
        total_shift = 0

    with pipe.progress_bar(total=len(timesteps) * num_frames // window_size) as progress:
        for i, t in enumerate(timesteps):
            progress.set_description(f"Diffusing timestep {t}...")

            if rotate:  # rotate latents by a random amount (so each timestep has different chunk borders)
                shift = shifts[i % len(shifts)]
                model_outputs = [None if pl is None else torch.roll(pl, shifts=shift, dims=2) for pl in model_outputs]
                latents = torch.roll(latents, shifts=shift, dims=2)
                total_shift += shift

            new_latents = torch.zeros_like(latents)
            new_outputs = torch.zeros_like(latents)

            for idx in range(0, num_frames, window_size):  # diffuse each chunk individually
                # update scheduler's previous outputs from our own cache
                pipe.scheduler.model_outputs = [model_outputs[(i - 1 - o) % order] for o in reversed(range(order))]
                pipe.scheduler.model_outputs = [
                    None if mo is None else mo[:, :, idx : idx + window_size, :, :].to(device)
                    for mo in pipe.scheduler.model_outputs
                ]
                pipe.scheduler.lower_order_nums = min(i, order)

                latents_window = latents[:, :, idx : idx + window_size, :, :].to(pipe.device)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_window] * 2) if do_classifier_free_guidance else latents_window
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents for scheduler
                pipe.scheduler.model_outputs = [
                    None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                    for mo in pipe.scheduler.model_outputs
                ]
                latents_window = rearrange(latents_window, "b c f h w -> (b f) c h w")
                noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

                # compute the previous noisy sample x_t -> x_t-1
                latents_window = pipe.scheduler.step(noise_pred, t, latents_window).prev_sample

                # reshape latents back for UNet
                latents_window = rearrange(latents_window, "(b f) c h w -> b c f h w", b=batch_size)

                # write diffused latents to output
                new_latents[:, :, idx : idx + window_size, :, :] = latents_window.cpu()

                # store scheduler's internal output representation in our cache
                new_outputs[:, :, idx : idx + window_size, :, :] = rearrange(
                    pipe.scheduler.model_outputs[-1], "(b f) c h w -> b c f h w", b=batch_size
                )

                progress.update()

            # update our cache with the further denoised latents
            latents = new_latents
            model_outputs[i % order] = new_outputs

    if rotate:
        new_latents = torch.roll(new_latents, shifts=-total_shift, dims=2)

    return new_latents


def get_audio_paths(df, testset='vggsound'):

    path = f'path/to/testset/{testset}'
    paths = []

    videos = set([file_path[:-4] for file_path in os.listdir(f"{path}/video/")])
    audios = set([file_path[:-4] for file_path in os.listdir(f"{path}/audio/")])
    samples = videos & audios

    df['ytid'] = df['ytid'].astype('str')
    ytids = set(df['ytid'].unique().tolist())
    for vid in list(samples):
        if vid[:11] in ytids:
            paths.append(os.path.join(f"{path}/audio/", vid + ".wav"))

    return paths


@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    window_size: Optional[int] = None,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    init_video: Optional[str] = None,
    init_weight: float = 0.5,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
    loop: bool = False,
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    with torch.autocast(device, dtype=torch.half):
        # prepare models
        pipe, temporal_token, local_windows, audio_token = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank)
        temporal_token = temporal_token.half().cuda()
        for i in range(len(local_windows)):
            if local_windows[i] is not None:
                local_windows[i] = local_windows[i].half().cuda()
        audio_token = audio_token.half().cuda() if audio_token is not None else None

        # prepare prompts
        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds, negative_prompt_embeds = compel(prompt), compel(negative_prompt) if negative_prompt else None

        # prepare input latents
        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            init_video=init_video,
            vae_batch_size=vae_batch_size,
        )
        init_weight = init_weight if init_video is not None else 0  # ignore init_weight as there is no init_video!

        # run diffusion
        latents = diffuse(
            pipe=pipe,
            latents=init_latents,
            init_weight=init_weight,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            rotate=loop or window_size < num_frames,
            temporal_token=temporal_token,
            local_windows=local_windows,
            audio_token=audio_token,
        )

        # decode latents to pixel space
        videos = decode(pipe, latents, vae_batch_size)

    return videos


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("--mapper_weights", type=str, required=True, help="The path to the embedder weights")
    parser.add_argument("-p", "--prompt", type=str, default="<temp> <local1> <local2> <local3> <local4> <class>", help="Text prompt to condition on")
    parser.add_argument("-n", "--negative-prompt", type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Directory to save output video to")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=384, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=384, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=24, help="Total number of frames to generate")
    parser.add_argument("-WS", "--window-size", type=int, default=None, help="Number of frames to process at once (defaults to full sequence). When less than num_frames, a round robin diffusion process is used to denoise the full sequence iteratively one window at a time. Must be divide num_frames exactly!")
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8, help="Batch size for VAE encoding/decoding to/from latents (higher values = faster inference, but more memory usage).")
    parser.add_argument("-s", "--num-steps", type=int, default=25, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=25, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-i", "--init-video", type=str, default=None, help="Path to video to initialize diffusion from (will be resized to the specified num_frames, height, and width).")
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5, help="Strength of visual effect of init_video on the output (lower values adhere more closely to the text prompt, but have a less recognizable init_video).")
    parser.add_argument("-f", "--fps", type=int, default=12, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-lP", "--lora_path", type=str, default="", help="Path to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lR", "--lora_rank", type=int, default=64, help="Size of the LoRA checkpoint's projection matrix (defaults to 64).")
    parser.add_argument("-rw", "--remove-watermark", action="store_true", help="Post-process the video with LAMA to inpaint ModelScope's common watermarks.")
    parser.add_argument("-l", "--loop", action="store_true", help="Make the video loop (by rotating frame order during diffusion).")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("--n", type=int, default=150, help="number of videos to generate")
    parser.add_argument("--testset", type=str, default='vggsound', help="dataset name")
    parser.add_argument("--audio_path", type=str, default='', help="path to audio file to run on")

    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    args.output_dir = f"{args.output_dir}/{args.model}"

    args.prompt = [args.prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size

    if args.window_size is None:
        args.window_size = args.num_frames

    if args.init_video is not None:
        vr = decord.VideoReader(args.init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args.num_frames, args.height, args.width), mode="trilinear")
        args.init_video = init

    # =========================================
    # ============= sample video =============
    # =========================================

    if args.audio_path:
        paths = [args.audio_path]

    else:
        df = pd.read_csv(f'datasets/{args.testset}.csv')
        df = df[df['set'] == 'test']
        paths = get_audio_paths(df, args.testset)

    for i, audio_path in enumerate(paths[:args.n]):

        ytid = audio_path.split('/')[-1][:-4]

        output_path = f"{args.output_dir}/{ytid}.mp4"
        output_audio = output_path.replace('mp4', 'wav')
        start_audio = 0
        end_audio = 2

        os.makedirs(args.output_dir, exist_ok=True)
        cut_wav_file(audio_path, output_audio[2:], start_audio, end_audio)

        videos = inference(
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            window_size=args.window_size,
            vae_batch_size=args.vae_batch_size,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            init_video=args.init_video,
            init_weight=args.init_weight,
            device=args.device,
            xformers=args.xformers,
            sdp=args.sdp,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            loop=args.loop,
        )

        # =========================================
        # ========= write outputs to file =========
        # =========================================

        for video in videos:
            if args.remove_watermark:
                print("Inpainting watermarks...")
                video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
                video = inpaint_watermark(video)
                video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

            else:
                video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

            video = video.byte().cpu().numpy()
            export_to_video(video, output_path, args.fps)

            combine_video_audio(output_path, output_audio)

            if os.path.exists(output_audio):
                os.remove(output_path)

            torch.cuda.empty_cache()

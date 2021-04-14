import torchvision.models as models
import numpy as np
import dataloader
import audio_video_utils
import deepdream
from utils import torch_utils

contentImgPath = "images/content/clemsim.png"
featureImgPaths = ["images/features/cats.jpg",
                   "images/features/cats-all.jpg",
                   "images/features/clouds.jpg",
                   "images/features/dogs.jpg",
                   "images/features/eyes-all.jpg",
                   "images/features/fleur.jpg",
                   "images/features/flowermacro.jpg",
                   "images/features/forest.jpg",
                   "images/features/highway.jpg",
                   "images/features/miel.jpg",
                   "images/features/headscanner.jpg",
                   "images/features/cats.jpg",
                   ]
musicPath = "musics/Cold Outside.mp3"


def generate_deep_dream_video(contentImgPath,
                              featureImgPaths,
                              musicPath,
                              outfilePath,
                              duration=None,
                              resolution=(124, 124),
                              frameLenght=1014
                              ):

    featureImgs = [dataloader.image_loader(path, resolution) for path in featureImgPaths]
    contentImg = dataloader.image_loader(contentImgPath, resolution)
    vgg = models.vgg19(pretrained=True).to(torch_utils.get_device()).eval()
    if duration:
        audio_video_utils.cut_song(musicPath, 0, duration)
    specm, gradm = audio_video_utils.generate_spec_vector(musicPath, 1024)
    chroma, chromasort = audio_video_utils.generate_chroma_vector(musicPath, 1024)
    classVec = audio_video_utils.generate_class_vector(chroma, chromasort)
    lr_vector = gradm/300+specm/300
    lr_vector[lr_vector < 2e-4] = 0
    lr_vector = np.convolve(lr_vector, np.ones((10,))/10, mode='same')
    frames = []
    for i in range(len(gradm)):
        print(i)
        frame = deepdream.run_deep_dream(resolution, contentImg, featureImgs[classVec[i]], vgg, 34, 40, lr=lr_vector[i] +0.0001)
        frames.append(frame)
    audio_video_utils.make_movie(frames, musicPath, outfilePath, frameLength=frameLenght)


generate_deep_dream_video(contentImgPath, featureImgPaths, musicPath,"videos/test.mp4")

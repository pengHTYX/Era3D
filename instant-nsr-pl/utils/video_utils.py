from moviepy.editor import VideoFileClip, VideoClip, ImageClip, CompositeVideoClip, ImageSequenceClip
from moviepy.editor import VideoFileClip, clips_array
import os
import argparse

def write_video(vclip, fps=25, save_path='res.mp4'):
    if isinstance(vclip, list):
        vclip = ImageSequenceClip(vclip, fps=fps)
    vclip.write_videofile(save_path, codec="libx264")

def load_video(vpath):
    return VideoFileClip(vpath)

def concat_video_clips(clips, videos_per_row=3):
    if isinstance(clips[0], str):
        clips = [VideoFileClip(v) for v in clips]
    elif not isinstance(clips[0], VideoFileClip):
        print(f'Find {len(clips)} clips')
    
    min_duration = min(clip.duration for clip in clips)
    clips = [clip.subclip(0, min_duration) for clip in clips]
    rows = [clips[i:i + videos_per_row] for i in range(0, len(clips), videos_per_row)]
    final_clip = clips_array(rows)
    return final_clip
    
def concat_video_and_frames(vpath, frames):
    vclip1 = VideoClip(vpath)
    vclip2 = ImageSequenceClip(frames, fps=25)
    clips = concat_video_clips([vclip1, vclip2])
    return clips

def concat_img_video(vclip, img):
    if isinstance(vclip, str):
        vclip = VideoFileClip(vclip)
    
    image_clip = ImageClip(img).set_duration(vclip.duration)
    image_clip = image_clip.resize(height=vclip.size[1])
    total_width = vclip.size[0] + image_clip.size[0]
    total_height = max(vclip.size[1], image_clip.size[1])
    composite_clip = CompositeVideoClip([
                                        image_clip.set_position('left'), vclip.set_position((image_clip.size[0], 0)), ],
                                        size=(total_width, total_height))
    return composite_clip


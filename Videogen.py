from Lib_imports import *
from pydub import AudioSegment

def VideoGen(Gitr, PrevItr, AudioPaths, BreakAudio, audioFile):
    init_frame = PrevItr 
    last_frame = Gitr 

    min_fps = 10
    max_fps = 120

    total_frames = last_frame-init_frame

    audio = AudioSegment.from_file(audioFile)
    lengthOfFile = audio.duration_seconds
    length = lengthOfFile + 1

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame,last_frame): 
        filename = f"steps/{i:04}.png"
        frames.append(Image.open(filename))

    fps = total_frames/length

  
    prompts = AudioPaths
    video_filename = ""
    if len(prompts) > 0:
     
        

        video_filename += audioFile.split(".")[0]
        video_filename += "__"

        video_filename += "_video"
        if BreakAudio :
            video_filename += "_BreakAudio"
        print("Video filename: "+ video_filename)

    video_filename = video_filename + ".mp4"
    
    from subprocess import Popen, PIPE
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', video_filename], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()

    print("Compressing video...")
    p.wait()
    print("Video ready")
    
    return video_filename
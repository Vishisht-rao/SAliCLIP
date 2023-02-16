import sys
sys.path.append('./taming-transformers')
from Lib_imports import *
from Parameters import *
from Models import *
from Data_pipeline import *
from Videogen import *
from Helpers import *
from Loadaudios import *
from moviepy.editor import *
import streamlit as st
from pydub import AudioSegment

from torchmetrics import AveragePrecision
import streamlit.components.v1 as components

from tempfile import TemporaryFile
from gtts import gTTS
import pickle

st.title("SAliCLIP Audio To Video Generator")
tab1, tab2 = st.tabs(["Video Generator", "Image Classification"])

def createStepsDirectory():
    if not os.path.isdir("steps"):
        os.mkdir("steps")

def audioInput(audio_files, BreakAudio, Brisque, step_size, cuts, mxitr):

        createStepsDirectory()
        
        genvideo = st.empty()
        genvideo.write("Generating Video....")
        my_bar = st.empty()
        my_bar.progress(0)
        prog_p = st.empty()
        prog_p.write("0%")
        tol = len(audio_files)
        final_clip = []
        MultipleAudiosDirectory = ""
        PrevPromptImage = ""
        Gitr = 0
        AudioPaths = audio_files

        PrevItr = 1
        for audioFile in AudioPaths:
            Gitr = GenSetup(audioFile, PrevPromptImage, Gitr, BreakAudio, Brisque, mxitr, step_size, cuts, my_bar, tol * mxitr, prog_p)
            videoFile = VideoGen(Gitr, PrevItr, AudioPaths, BreakAudio, audioFile)

            if BreakAudio:
                os.system("rm " + MultipleAudiosDirectory + audioFile.split(".")[0] + "_1.wav")
                os.system("rm " + MultipleAudiosDirectory + audioFile.split(".")[0] + "_2.wav")

            PrevItr = Gitr
            video_clip = VideoFileClip(videoFile)
            audio_clip = AudioFileClip(MultipleAudiosDirectory + audioFile)
            print(MultipleAudiosDirectory + audioFile)
            audio_clip = audio_clip.volumex(1.25)
            final_clip.append(video_clip.set_audio(audio_clip))


        final = concatenate_videoclips(final_clip)    
        final_name = AudioPaths[0][:-4] + '_' + AudioPaths[-1][:-4] + '___video.mp4'

        try:
            final.write_videofile(final_name, 
                         codec='libx264', 
                         audio_codec='aac', 
                         temp_audiofile='temp-audio.m4a', 
                         remove_temp=True
                         )
            genvideo.empty()
            my_bar.write("Video is Ready")
            prog_p.write("Displaying Video....")
        except Exception as e:
            print("Exception Occured : ", e)
            quit()

        return final_name

with tab1:
    

    FinalVid = ""
    clicked = False
    cnt = 0
    uploaded_files = st.file_uploader("Choose audio files", accept_multiple_files=True)
    audio_files = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        cnt += 1
        st.audio(bytes_data)

        file_var = AudioSegment.from_wav(uploaded_file)
        file_var.export("aud" + str(cnt) + ".wav", format = "wav")
        audio_files.append("aud" + str(cnt) + ".wav")

    BreakAudio = st.checkbox("BreakAudio")
    Brisque = st.checkbox("Brisque")
    step_size = st.slider("Select the step size", 0.0, 1.0, 0.1)
    cuts = st.slider("Select the number of cuts to perform on the image", 2, 128, 64)
    mxitr = st.number_input('Enter number of iterations', min_value = 10, value = 50)
    clicked = st.button("Generate Video")
    if clicked:
        FinalVid =  audioInput(audio_files, BreakAudio, Brisque, step_size, cuts, mxitr)

    if FinalVid != "":
        for num in range(1, cnt + 1):
            os.system("rm aud" + str(num) + ".wav")
            if BreakAudio:
                os.system("rm aud" + str(num) + "___video_BreakAudio.mp4")
            else:
                os.system("rm aud" + str(num) + "___video.mp4")
                
        
        video_file = open(FinalVid, 'rb')

        video_bytes = video_file.read()

        st.video(video_bytes)
        

with tab2:
        
    upload_file = st.file_uploader("upload an image for classification", accept_multiple_files=False)
    
    if os.path.isfile("imageinput.png"):
        os.system("rm imageinput.png")
        
    if upload_file != None:
    
        loadAudiofileName = 'loadAudios'
        
        if device == "cpu":
            loadAudiofileName += "CPU"
            
        loadAudiofile = open(loadAudiofileName, 'rb')     
        loadAudios = pickle.load(loadAudiofile)
        loadAudiofile.close()
        
        
        EncodedAudiosNorm = loadAudios["EncodedAudiosNorm"]
        AudioClasses = loadAudios["AudioClasses"]
        AudioClassCounts = loadAudios["AudioClassCounts"]
        
        if device == "cpu":
            EncodedAudiosNorm = EncodedAudiosNorm.type(torch.float32)
            
        bytes_data = upload_file.read()

        file_name = upload_file.name
        st.image(bytes_data)

        with open("imageinput.png", 'wb') as f:
            f.write(bytes_data)

        image = preprocessor._processImageFile("imageinput.png")
        image = image.unsqueeze(0).to(device)

        EncodedImage = SAliCLIP.ImageEncoder(image)
        if SAliCLIP.hparams.UseImageProjectionHead:
            EncodedImage = SAliCLIP.ImageHead(EncodedImage)

        EncodedImageNorm = nn.functional.normalize(EncodedImage, p=2, dim=1)
       
        similarities =  EncodedImageNorm @ EncodedAudiosNorm.T
        similarities = similarities.squeeze(0)

        similarities_per_class = []
        cursum = 0
        for i in range(len(AudioClasses)):
            similarities_per_class.append(torch.max(similarities[cursum:cursum+AudioClassCounts[i]]).item())
            cursum += AudioClassCounts[i]

        similarities_per_class = np.array(similarities_per_class)
        similarity_dict = {k:(v + 1) * 50 for v, k in zip(similarities_per_class, AudioClasses)}

        Keymax = max(zip(similarity_dict.values(), similarity_dict.keys()))[1]

        prompt = "A photo of a " + ' '.join(Keymax.split("_"))
        
        tts = gTTS(text=prompt, lang='en')
        tempFile = TemporaryFile()
        tts.write_to_fp(tempFile)
        tempFile.seek(0)
        
        audio_bytes = tempFile.read()

        st.audio(audio_bytes, format='audio/mp3')
        st.subheader("The image is classified as : " + ' '.join(Keymax.split("_")))

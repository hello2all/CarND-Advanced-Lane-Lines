# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from pipeline import Pipe_line

pipe_line = Pipe_line()
white_output = 'lane1.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipe_line.process) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
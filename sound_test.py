# pip install gTTS
# pip install playsound

from gtts import gTTS
import playsound
def createSound(text='กรุณาสวมแมสก์', lang='th', filename='wearmask.mp3'):
	""" text='กรุณาสวมแมสก์', lang='th', filename='wearmask.mp3' """
	tts=gTTS(text=text,lang=lang)
	tts.save(filename)

# createSound('ขอบคุณที่สวมแมสก์', 'th', 'thxwarmask.mp3')

# playsound.playsound('wearmask.mp3', True)
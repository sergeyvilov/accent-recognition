import os
import uuid
from flask import Flask, flash, request, redirect,render_template, url_for
from subprocess import Popen,PIPE
from accents_ai import EnsemblePredictor

UPLOAD_FOLDER = 'files'
max_duration = '02:00' #https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "ytcM97QjutUL"

#models_dir = '/Users/sergey.vilov/Documents/my_projects/accents/sprec/speech_accent_archive/nnc_logs/single_true/5_fold/no_effect_production/'
models_dir = './models/'
accent_predictor = EnsemblePredictor(models_dir, n_models=10)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/process-record', methods=['POST'])
def save_record():
    #process the sendData function result
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    command = f"ffmpeg -t {max_duration} -i pipe: -loglevel error -ar 16000 -f wav pipe:1" #convert to wav with sampling rate of 16kHz
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)

    process.stdin.write(file.read())
    process.stdin.close()
    waveform = process.stdout.read()
    errmsg = process.stderr.read().decode('utf-8')

    #print(waveform)

    #waveform=subprocess.check_output(f"ffmpeg -y -nostdin -hide_banner -loglevel error -i {path} -ar 16000 -f wav -to 00:00:{self.max_duration} pipe:1", shell=True).rstrip()
    #file_name = str(uuid.uuid4()) + ".webm"
    #full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    #file.save(full_file_name)

    accent_probs = accent_predictor(waveform) #perform inference

    return accent_probs


if __name__ == '__main__':
    app.run()

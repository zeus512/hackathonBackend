

STEPS TO RUN :

rm  -rf  .venv

python -m venv .venv

source .venv/bin/activate

pip install Flask google-cloud-speech google-cloud-translate google-cloud-texttospeech google-cloud-storage
pip install spacy   
python -m spacy download en_core_web_sm
pip install pandas                     
pip install spacy-lookups-data   
pip install --upgrade google-cloud-aiplatform

python app.py --host=0.0.0.0 
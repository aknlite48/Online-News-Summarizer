# Setting up virtual environment
python3 -m venv venv \
source venv/bin/activate \

# Install packages and launch program
pip install -r requirements.txt \
python app.py \

# Closing virtual environment
deactivate

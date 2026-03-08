echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

echo "Environment ready"
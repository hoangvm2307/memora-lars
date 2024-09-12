from dotenv import load_dotenv
from routes import app

load_dotenv()

if __name__ == "__main__":
    # main()
    app.run(debug=True)

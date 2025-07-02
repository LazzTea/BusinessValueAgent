# BusinessValueAgent

A simple program designed to use AI Agents and Algorithms to predict the value of a business based on information
collected from the NASDAQ and yfinancial's data bases.  The program asks you what company you want to know about and than
responds based on the program designed.


# How to use the program!

Clone the Github
```bash
git clone https://github.com/LazzTea/BusinessValueAgent.git
cd BusinessValueAgent
```

Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the requirements
```bash
pip install -r requirements.txt
```

Create and .env file and add this here with the key being your api key
```bash
OPENAI_API_KEY=your_openai_key_here
```

Run the program
```bash
python main.py
```
Follow along with the program and deactivate venv when finished
```bash
deactivate
```

# Potential Improvements

Minor Improvements that the program can almost already do or would take a short time to add,
not added due to time restraint

- Use k-Fold cross validation to optimize hyperparamaters, specifically max_depth which is likely holding back the program currently
- Have the program be able to check for a new company if the user gives the NASDAQ data, the program can do this but the AI Agent currently
does not implement the specific function but techinically the algorithm already can check for this
- Expand the test data, not done because the data takes a while to train already

Major Improvements that the program cannot do at all right now but would relate to what it already does.
- Check for the value of a company based on different metrics such as through common opinion like tweets and articles.
- Analyse the stock value of these companies
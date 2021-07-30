from flask import Flask, request, render_template
import model

app = Flask(__name__)


@app.route('/question', methods=['GET', 'POST'])
def question_page():
    return render_template('question.html')


@app.route('/answer', methods=['POST'])
def answer_page():
    if request.method == 'POST':
        ans = model.run_model_qa(request.form['doc'], request.form['question'])
        return render_template('answer.html', ans=ans)
    else:
        print('Invalid Request')


app.run()

from flask import Flask, request, render_template
import model

app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/question', methods=['POST'])
def question_page():
    if request.method in ['GET', 'POST']:
        return render_template('question.html')
    else:
        print('Invalid Request')


@app.route('/answer', methods=['POST'])
def answer_page():
    if request.method == 'POST':
        answer = model.run_model_qa(request.form['context'], request.form['question'])
        return render_template('answer.html', answer=answer)
    else:
        print('Invalid Request')


if __name__ == '__main__':
    app.run(debug=False)

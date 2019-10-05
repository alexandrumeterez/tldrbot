from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import sys
from models.BERTtldr import BERTtldr

model = BERTtldr('./model')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
socketio = SocketIO(app)


@app.route('/')
def sessions():
    return render_template('session.html')


@socketio.on('sendQuestionAndDocText')
def handleSendQuestionAndDocText(json, methods=['GET', 'POST']):
    if json['type'] == 'log':
        print('[LOG] ' + str(json))
    if json['type'] == 'data':
        questionText = json['questionText']
        docText = json['docText']
        prediction = model.predict(docText, questionText)
        answer = prediction['answer']
        if len(answer) == 0:
            answer = 'I don\'t know'
        print('[LOG] Answer: {}'.format(answer))
        socketio.emit('sendAnswer', answer)


if __name__ == '__main__':
    socketio.run(app, debug=True)

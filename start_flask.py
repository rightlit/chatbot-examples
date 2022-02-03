from flask import Flask
app = Flask(__name__)

# 일반적인 라우트 방식입니다.
@app.route('/hello')
def board():
    return "Hello, World"

# URL 에 매개변수를 받아 진행하는 방식입니다.
@app.route('/board/<article_idx>')
def board_view(article_idx):
    return article_idx

# 위에 있는것이 Endpoint 역활을 해줍니다.
@app.route('/boards',defaults={'page':'index'})
@app.route('/boards/<page>')
def boards(page):
    return page+"페이지입니다."

app.run(host="localhost",port=5001)

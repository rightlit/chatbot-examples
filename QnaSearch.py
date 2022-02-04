# 클래스 선언 
class QnaSearch:

    # 속성 생성 
    def __init__(self, name, age, address): 
        self.name = name 
        self.age = age 
        self.address = address 

    # 메소드 생성 
    def info(self): 
        print('저의 이름은 {0}이고, 나이는 {1}, 사는곳은 {2} 입니다'.format(self.name, self.age, self.address)) 

# 인스턴스 생성 
qna = QnaSearch('nirsa', 80, '인천 광역시') 
# 메소드 호출 
qna.info()

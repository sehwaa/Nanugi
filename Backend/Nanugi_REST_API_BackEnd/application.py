from flask import Flask, jsonify, session, make_response, request
from flask_restful import reqparse, abort, Api, Resource
import pymysql
import json
import os
from flaskext.mysql import MySQL
from flask_cors import CORS
from flask_session import *
#from redis import Redis
from uuid import uuid4
from flask_s3 import FlaskS3
from boto.s3.connection import S3Connection
from boto.s3.key import Key as S3Key
from werkzeug.datastructures import FileStorage
from io import StringIO,BytesIO
import io
import boto3
import pandas as pd
import itertools
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.extmath import randomized_svd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import operator
# nltk.download('punkt') # exception 구문으로 전환하기
import re

#mysql 객체
mysql = MySQL(cursorclass=pymysql.cursors.DictCursor)

#Flask 인스턴스 생성
application = Flask(__name__)
api = Api(application)
application.secret_key = "\xaa6\xe06n\xc5\xd8=/S3d\xe7\xe5N\xfe%\xa92Ojm\x80\xf5"
CORS(application, supports_credentials=True, resources=r'/*')

application.config['MYSQL_DATABASE_USER'] = 'root'
application.config['MYSQL_DATABASE_PASSWORD'] = 'rootdb5jo'
application.config['MYSQL_DATABASE_DB'] = 'nanugi'
application.config['MYSQL_DATABASE_HOST'] = 'nanugi.cnwhswqzcfic.ap-northeast-2.rds.amazonaws.com'
application.config['FLASK3_BUCKET_NAME'] = 'elasticbeanstalk-ap-northeast-2-164387737785'
application.config.from_object(__name__)

application.config['ALLOWED_EXTENSIONS'] = ['jpg', 'jpeg', 'png']
application.config['FILE_CONTENT_TYPES'] = {
    'jpg' : 'image/jpeg',
    'jpeg' : 'image/jpeg',
    'png' : 'image/png'
}
application.config['AWS_ACCESS_KEY_ID'] = 'AKIAJYUUY7OHL46KXSPA'
application.config['AWS_SECRET_ACCESS_KEY'] = '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'
s3 = FlaskS3(application)

mysql.init_app(application)

# load model as global from s3
bucket='sagemaker-test02'
data_key = 'model'
data_location = 's3://{}/{}'.format(bucket, data_key)
s3 = boto3.resource('s3', aws_access_key_id=application.config['AWS_ACCESS_KEY_ID'], aws_secret_access_key=application.config['AWS_SECRET_ACCESS_KEY'])
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_tdmvector.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_tdm_model = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_tfidf.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_tfidf_model = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_svm_b.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_svm_model = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_svm_m.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_svm_model1 = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_svm_s.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_svm_model2 = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("model/model_svm_bms.sav", data)
    data.seek(0)    # move back to the beginning after writing
    loaded_svm_model3 = joblib.load(data)
with BytesIO() as data:
    s3.Bucket(bucket).download_fileobj("json_category/cate1.json", data)
    data.seek(0)    # move back to the beginning after writing
    json_category = pd.read_json(data)



# 1)define hangule function
# target = "컵 핀 개 관 카드 캐시백 나이키 나이키 565656 @@# abf " # sample
stop_words = []
def hanguel_func(target) :
    # 1) 한글만 남기기
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub(' ', target) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

    # 2) 불필요한 단어 제거
    stop_words = "카드 캐시백 무료 배송 무료배송 당일발송 무료견적 청구할인 천원 세트 케이스포함 신한 이월상품 일반용 포함 일반 "
    stop_words += "현대 현대백화점 롯데백화점 신세계백화점 백화점 하프클럽 갤러리아 행사 수원점 플라자 신세계인천점 신세계센텀점 "
    stop_words += "나비 환영 여성 남성 성인용 추가 플러스 인용 개월 할인 "
    stop_words += "디자인 컬러 검정 블랙 레드 빨강 옐로우 노랑 실버 네이비 그린 블루 "
    stop_words=stop_words.split(' ')
    # word_tokens = word_tokenize(result)
    word_tokens = result.split(' ')
    result_2 = []
    for w in word_tokens:
        if w not in stop_words:
            result_2.append(w)

    # 3) 중복 워드 없애기
    result_2 = list(set(result_2))

    # 4) 한글자 미만 줄이기
#     save_words = "컵 핀 넥 퀸 젤 캡 볼 힐 백 꽃 빔 꿀 폰 옷 천 면 찜 흙 껌 솥 떡"
    save_words = " "
    save_words = save_words.split(' ')
    result_3 = []
    for i in range(len(result_2)) :
        if len(result_2[i]) == 1:
            if result_2[i] in list(save_words) :
                result_3.append(result_2[i])
            else:
                pass
        else:
            result_3.append(result_2[i])
#     print(result_3)

    # 5) 형용사 없애기(형태소 분석)

    # 9) 최종 전처리 결과 return
    answer = ""
    for i in range (0, len(result_3)):
        answer += result_3[i] + " "

    print(answer)
    return answer
# hanguel_func(target) # check

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# accuarcy weight function
def weigth_pct(target) :
    weigth_pct_list = []
    count = 0
    x = np.sort(target)[::-1] 
    for i in x :
        weigth_pct_list.append(round(i*(len(x)-count), 2))
        count += 1
    weigth_pct_list = weigth_pct_list/sum(weigth_pct_list)
    return weigth_pct_list

#s3 파일 업로드
def upload_s3(file, key_name, content_type, bucket_name):
    #s3 connection
    conn = S3Connection(application.config['AWS_ACCESS_KEY_ID'], application.config['AWS_SECRET_ACCESS_KEY'], host='s3.ap-northeast-2.amazonaws.com')

    #upload the file after getting the right bucket
    bucket = conn.get_bucket(bucket_name)
    obj = S3Key(bucket)
    obj.name = key_name
    obj.content_type = content_type
    obj.set_contents_from_string(file.getvalue())
    obj.set_acl('public-read')

    #close stringio object
    file.close()

    return obj.generate_url(expires_in=0, query_auth=False)

# class FileStorageArgument(reqparse.Argument):
#     def convert(self, value, op):
#         if self.type is FileStorage:
#             return value

#         super(FileStorageArgument, self).convert(*args, **kwargs)

#엔드유저 신규상품 등록 요청
class UploadImage(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('idx')
            parser.add_argument('product_name')
            parser.add_argument('product_price')
            parser.add_argument('product_photo', required=True, type=FileStorage, location='files')
            args = parser.parse_args()
            product_photo = args['product_photo']

            if args['idx'] == None or args['product_name'] == None or args['product_price'] == None or args['product_photo'] == None:
                return 'failed', 400
            else:
                extension = product_photo.filename.rsplit('.', 1)[1].lower()
                if '.' in product_photo.filename and not extension in application.config['ALLOWED_EXTENSIONS']:
                    abort(400, message="File extension is not one of our supported types.")

            #create a file object of the image
            image_file = BytesIO()
            product_photo.save(image_file)

            #upload to s3
            key_name = '{0}.{1}'.format(args['idx']+"-"+args['product_name'].replace('/', ''), extension)
            content_type = application.config['FILE_CONTENT_TYPES'][extension]
            bucket_name = application.config['FLASK3_BUCKET_NAME']
            image_url = upload_s3(image_file, key_name, content_type, bucket_name)

            # down load the models from s3
            ## setting s3's model path
            product_name_pre = hanguel_func(args['product_name'])
            #--------------------------------------------------------
                        # make sample test
            target=0
            X_test = []
            X_test.append(product_name_pre)
            ## (1) Tdm vector
            X_test_tdm = loaded_tdm_model.transform(X_test)### test (transform)
            ## (2) TF_IDF
            tfidfv_test = loaded_tfidf_model.transform(X_test_tdm) #### test (transform)
            ## (3) svm modeling
            # if str(loaded_svm_model.predict(tfidfv_test[target])[0]) == bms_category.split('_')[0] :
            if str(loaded_svm_model.predict(tfidfv_test[target])[0]) == str(loaded_svm_model3.predict(tfidfv_test[target])[0].split('_')[0]) :
                # using bms one prediction
                # bms predict one
                bms_category = str(loaded_svm_model3.predict(tfidfv_test[target])[0])
                b_category = bms_category.split('_')[0]
                m_category = bms_category.split('_')[1]
                s_category = bms_category.split('_')[2]
            else :
                # using each prediction
                b_category = str(loaded_svm_model.predict(tfidfv_test[target])[0])
                m_category = str(loaded_svm_model1.predict(tfidfv_test[target])[0])
                s_category = str(loaded_svm_model2.predict(tfidfv_test[target])[0])

            #--------------------------------------------------------
            # # make sample test
            # X_test = []
            # X_test.append(product_name_pre)

            # # test

            # X_test_tdm = loaded_tdm_model.transform(X_test)### test (transform)
            # # print(X_train_tdm.shape)
            # ## (2) TF_IDF

            # tfidfv_test = loaded_tfidf_model.transform(X_test_tdm) #### test (transform)
            # # get percentage

            # target=0
            # scores = loaded_svm_model.decision_function(tfidfv_test[target])
            # # print('input :',X_test[target])
            # # print('predicted value :',loaded_svm_model.predict(tfidfv_test[target]))
            # # print('real value :',y_test)
            # # print('percentage :', softmax(scores).max())

            # b_category = str(loaded_svm_model.predict(tfidfv_test[target])[0])
            # m_category = str(loaded_svm_model1.predict(tfidfv_test[target])[0])
            # s_category = str(loaded_svm_model2.predict(tfidfv_test[target])[0])

            #--------------------------------------------------------------
            scores = loaded_svm_model.decision_function(tfidfv_test[target])
            weigth_pct_list = weigth_pct(softmax(scores[0]))
            for i in range(15) : 
                weigth_pct_list = weigth_pct(weigth_pct_list)
            accuarcy = str(int(round(weigth_pct_list[0],2)*100))+'%'
            
            # conn = mysql.connect()
            # cur = conn.cursor()
            # sql = "INSERT INTO enduser_product (idx, product_name, product_price, product_photo, b_category, m_category, s_category, accuarcy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            # data =(args['idx'], args['product_name'], args['product_price'], image_url, b_category, m_category, s_category, accuarcy)
            # cur.execute(sql, data)
            # conn.commit()
            # sql2 = "SELECT product_idx, b_category, m_category, s_category FROM enduser_product WHERE product_photo=%s"
            # cur.execute(sql2, image_url)
            # result = cur.fetchone()
            # conn.close()

            # print(df_1[df_1['bcateid'] == loaded_model.predict(tfidfv_test[target])[0]].head())
            # # print(softmax(scores))
            # softmax(scores).sum()


            # change number to string
            result_b = json_category['b'][json_category['b'] == int(b_category)].index[0]
            result_m = json_category['m'][json_category['m'] == int(m_category)].index[0]
            result_s = json_category['s'][json_category['s'] == int(s_category)].index[0]

            result_final = dict(zip(('msg', 'status', 'b_category', 'm_category', 's_category', 'accurcy'), ('ok', 200, result_b, result_m, result_s, accuarcy)))
            json_data = json.dumps(result_final, ensure_ascii=False)
            res = make_response(json_data)
            res.headers['Content-Type'] = 'application/json'

            return res
        except Exception as e:
            return e, 500

#엔드유저 로그인
class signin(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        parser.add_argument('pw')
        args = parser.parse_args()

        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM enduser_member WHERE id=%s"
        cur.execute(sql, args['id'])

        try:
            result = cur.fetchone()
            if result['pw'] == args['pw']:
                session['idx'] = result['idx']
                session['name'] = result['enduser_name']
                session['id'] = result['id']
                data = dict(zip(('msg', 'status', 'session_info'), ('ok', 200, json.dumps(dict(zip(('idx', 'name', 'id'),(session['idx'], session['name'], session['id'])))))))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 200)
                res.headers['Content-Type'] = 'application/json'
                return res
            else:
                data = dict(zip(('msg', 'status'), ('비밀번호 틀림', 401)))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 401)
                res.headers['Content-Type'] = 'application/json'
                return res
        except Exception as e:
            data = dict(zip(('msg', 'status'), (e, 401)))
            json_data = json.dumps(data, ensure_ascii=False)
            res = make_response(json_data, 401)
            res.headers['Content-Type'] = 'application/json'
            return res

#엔드유저 로그아웃
class signout(Resource):
    def post(self):
        session.clear()
        return 'success',200

#엔드유저 카테고리 수정요청
class modify(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('idx')
            parser.add_argument('product_name')
            parser.add_argument('product_price')
            parser.add_argument('cate_b')
            parser.add_argument('cate_m')
            parser.add_argument('cate_s')
            parser.add_argument('accurcy')
            parser.add_argument('product_photo', required=True, type=FileStorage, location='files')
            args = parser.parse_args()
            product_photo = args['product_photo']

            if args['idx'] == None or args['product_name'] == None or args['product_price'] == None or args['product_photo'] == None or args['cate_b'] == None or args['cate_m'] == None or args['cate_s'] == None or args['accurcy'] == None:
                return 'failed', 400
            else:
                extension = product_photo.filename.rsplit('.', 1)[1].lower()
                if '.' in product_photo.filename and not extension in application.config['ALLOWED_EXTENSIONS']:
                    abort(400, message="File extension is not one of our supported types.")

            #create a file object of the image
            image_file = BytesIO()
            product_photo.save(image_file)

            #upload to s3
            key_name = '{0}.{1}'.format(args['idx']+"-"+args['product_name'].replace('/', ''), extension)
            content_type = application.config['FILE_CONTENT_TYPES'][extension]
            bucket_name = application.config['FLASK3_BUCKET_NAME']
            image_url = upload_s3(image_file, key_name, content_type, bucket_name)

            conn = mysql.connect()
            cur = conn.cursor()
            sql = "INSERT INTO enduser_product (idx, product_name, product_price, product_photo, complete, b_category, m_category, s_category, accuarcy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data =(args['idx'], args['product_name'], args['product_price'], image_url, "N", args['cate_b'], args['cate_m'], args['cate_s'], args['accurcy'])
            cur.execute(sql, data)
            conn.commit()
            conn.close()

            return 'success',200

        except Exception as e:
            return e, 500

#엔드유저 카테고리 최종 등록요청
class register(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('idx')
            parser.add_argument('product_name')
            parser.add_argument('product_price')
            parser.add_argument('cate_b')
            parser.add_argument('cate_m')
            parser.add_argument('cate_s')
            parser.add_argument('accurcy')
            parser.add_argument('product_photo', required=True, type=FileStorage, location='files')
            args = parser.parse_args()
            product_photo = args['product_photo']

            if args['idx'] == None or args['product_name'] == None or args['product_price'] == None or args['product_photo'] == None or args['cate_b'] == None or args['cate_m'] == None or args['cate_s'] == None or args['accurcy'] == None:
                return 'failed', 400
            else:
                extension = product_photo.filename.rsplit('.', 1)[1].lower()
                if '.' in product_photo.filename and not extension in application.config['ALLOWED_EXTENSIONS']:
                    abort(400, message="File extension is not one of our supported types.")

            #create a file object of the image
            image_file = BytesIO()
            product_photo.save(image_file)

            #upload to s3
            key_name = '{0}.{1}'.format(args['idx']+"-"+args['product_name'].replace('/', ''), extension)
            content_type = application.config['FILE_CONTENT_TYPES'][extension]
            bucket_name = application.config['FLASK3_BUCKET_NAME']
            image_url = upload_s3(image_file, key_name, content_type, bucket_name)

            conn = mysql.connect()
            cur = conn.cursor()
            sql = "INSERT INTO enduser_product (idx, product_name, product_price, product_photo, complete, b_category, m_category, s_category, accuarcy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data =(args['idx'], args['product_name'], args['product_price'], image_url, "Y", args['cate_b'], args['cate_m'], args['cate_s'], args['accurcy'])
            cur.execute(sql, data)
            conn.commit()
            conn.close()

            return 'success',200

        except Exception as e:
            return e, 500

#엔드유저 회원가입
class signup(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        parser.add_argument('pw')
        parser.add_argument('enduser_name')
        args = parser.parse_args()

        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM enduser_member WHERE id=%s"
        check = cur.execute(sql, args['id'])

        if check == 1:
            return '중복된 id', 409
        else :
            if args['id'] == None or args['pw'] == None or args['enduser_name'] == None:
                return 'failed', 400
            else:
                sql2 = "INSERT INTO enduser_member (id,pw,enduser_name) VALUES (%s,%s,%s)"
                cur.execute(sql2, (args['id'], args['pw'], args['enduser_name']))
                conn.commit()
                conn.close()
                return 'success', 200

#엔드유저 회원가입 시 중복 체크
class duplicatecheck(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        args = parser.parse_args()

        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT id FROM enduser_member WHERE id=%s"
        check = cur.execute(sql, args['id'])

        if check == 1 :
            return '중복된 id', 409
        else:
            return '사용가능한 id', 200

#엔드유저 등록 요청 상품 조회
class check_n(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('idx')
        args = parser.parse_args()

        if args['idx'] == None:
            return 'failed', 400
        else:
            conn = mysql.connect()
            cur = conn.cursor()
            sql = "SELECT * FROM enduser_product WHERE final_complete='N' and idx=%s order by product_idx desc"
            cur.execute(sql, args['idx'])
            try:
                result = cur.fetchall()

                product = []
                for i in result:
                    data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'product_photo','b_category', 'm_category', 's_category'),(i['product_idx'], i['product_name'], i['accuarcy'], i['product_photo'], i['b_category'], i['m_category'], i['s_category'])))
                    product.append(data)
                    # data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'product_photo','b_category', 'm_category', 's_category'),(i['product_idx'], i['product_name'], i['accuarcy'], i['product_photo'], i['b_category'], i['m_category'], i['s_category'])))
                    # product.append(json.dumps(data, ensure_ascii=False))

                result_final = dict(zip(('msg', 'status', 'product'), ('ok', 200, product)))
                json_data = json.dumps(result_final, ensure_ascii=False)
                res = make_response(json_data)
                res.headers['Content-Type'] = 'application/json'
                
                return res
            except Exception as e:
                return e, 401

#엔드유저 등록 완료 상품 조회
class check_y(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('idx')
        args = parser.parse_args()

        if args['idx'] == None:
            return 'failed', 400
        else:
            conn = mysql.connect()
            cur = conn.cursor()
            sql = "SELECT * FROM enduser_product WHERE final_complete='Y' and idx=%s order by product_idx desc"
            cur.execute(sql, args['idx'])
            try:
                result = cur.fetchall()

                product = []
                for i in result:
                    data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'product_photo','b_category', 'm_category', 's_category'),(i['product_idx'], i['product_name'], i['accuarcy'], i['product_photo'], i['b_category'], i['m_category'], i['s_category'])))
                    product.append(data)
                    # data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'product_photo','b_category', 'm_category', 's_category'),(i['product_idx'], i['product_name'], i['accuarcy'], i['product_photo'], i['b_category'], i['m_category'], i['s_category'])))
                    # product.append(json.dumps(data, ensure_ascii=False))

                result_final = dict(zip(('msg', 'status', 'product'), ('ok', 200, product)))
                json_data = json.dumps(result_final, ensure_ascii=False)
                res = make_response(json_data)
                res.headers['Content-Type'] = 'application/json'
                
                return res
            except Exception as e:
                return e, 401

#관리자 로그인
class admin_signin(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        parser.add_argument('pw')
        args = parser.parse_args()

        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM admin_member WHERE id=%s"
        cur.execute(sql, args['id'])

        try:
            result = cur.fetchone()
            if result['pw'] == args['pw']:
                session['idx'] = result['idx']
                session['name'] = result['enduser_name']
                session['id'] = result['id']
                data = dict(zip(('msg', 'status', 'session_info'), ('ok', 200, json.dumps(dict(zip(('idx', 'name', 'id'),(session['idx'], session['name'], session['id'])))))))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 200)
                res.headers['Content-Type'] = 'application/json'
                return res
            else:
                data = dict(zip(('msg', 'status'), ('비밀번호 틀림', 401)))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 401)
                res.headers['Content-Type'] = 'application/json'
                return res
        except Exception as e:
            data = dict(zip(('msg', 'status'), (e, 401)))
            json_data = json.dumps(data, ensure_ascii=False)
            res = make_response(json_data, 401)
            res.headers['Content-Type'] = 'application/json'
            return res

#관리자 로그아웃
class admin_signout(Resource):
    def post(self):
        session.clear()
        return 'success',200

#등록 대기 상품 조회
class waitproduct(Resource):
    def get(self):
        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM enduser_product WHERE final_complete='N'"
        cur.execute(sql)

        try:
            result = cur.fetchall()

            product = []
            for i in result:
                product_idx = i['product_idx']
                product_name = i['product_name']
                accuarcy = i['accuarcy']
                product_photo = i['product_photo']
                category = dict(zip(('b_category', 'm_category', 's_category', 'd_category'),(i['b_category'], i['m_category'], i['s_category'], i['d_category'])))
                data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'product_photo','category'),(product_idx, product_name, accuarcy, product_photo, category)))
                product.append(json.dumps(data, ensure_ascii=False))

            json_dict = dict()
            json_dict['product'] = product
            res = make_response(json.dumps(json_dict), 200)
            res.headers['Content-Type'] = 'application/json'
            return res
        except Exception as e:
            return e, 401

#분류 정확도 일정 임계치 이상 상품 조회
class accurateproduct(Resource):
    def get(self):
        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM enduser_product WHERE accuarcy >= 75 and final_complete = 'N' and complete = 'Y'"
        cur.execute(sql)

        try:
            result = cur.fetchall()

            product = []
            for i in result:
                product_idx = i['product_idx']
                product_name = i['product_name']
                accuarcy = i['accuarcy']
                category = dict(zip(('b_category', 'm_category', 's_category', 'd_category'),(i['b_category'], i['m_category'], i['s_category'], i['d_category'])))
                data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'category'),(product_idx, product_name, accuarcy, category)))
                product.append(json.dumps(data, ensure_ascii=False))

            json_dict = dict()
            json_dict['product'] = product
            res = make_response(json.dumps(json_dict), 200)
            res.headers['Content-Type'] = 'application/json'
            return res
        except Exception as e:
            return e, 401

#분류 정확도 일정 임계치 미만 or 판매자가 수정 요청한 상품 조회
class inaccurateproduct(Resource):
    def get(self):
        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM enduser_product WHERE (accuarcy < 75 and final_complete = 'N') or complete = 'N'"
        cur.execute(sql)

        try:
            result = cur.fetchall()

            product = []
            for i in result:
                product_idx = i['product_idx']
                product_name = i['product_name']
                accuarcy = i['accuarcy']
                category = dict(zip(('b_category', 'm_category', 's_category', 'd_category'),(i['b_category'], i['m_category'], i['s_category'], i['d_category'])))
                data = dict(zip(('product_idx', 'product_name', 'accuarcy', 'category'),(product_idx, product_name, accuarcy, category)))
                product.append(json.dumps(data, ensure_ascii=False))

            json_dict = dict()
            json_dict['product'] = product
            res = make_response(json.dumps(json_dict), 200)
            res.headers['Content-Type'] = 'application/json'
            return res
        except Exception as e:
            return e, 401

class temp(Resource):
    def post(self):
        s3 = boto3.resource('s3', aws_access_key_id=application.config['AWS_ACCESS_KEY_ID'], aws_secret_access_key=application.config['AWS_SECRET_ACCESS_KEY'])
        content_object = s3.Object('sagemaker-recommendsystem', 'CDs_and_Vinyl_5.json')
        file_content = content_object.get()['Body'].read().decode('utf-8')

        X = pd.read_json(file_content, lines=True)
        asin_list = X['asin']
        overall_list = X['overall']
        reviewerID_list = X['reviewerID']
        category= 'CDs_and_Vinyl_5'

        conn = mysql.connect()
        cur = conn.cursor()

        for i in range(2, len(asin_list)):
            sql = "INSERT INTO amazonrating (asin,overall, reviewerID, category) VALUES (%s, %s, %s, %s)"
            cur.execute(sql, (str(asin_list[i]), str(overall_list[i]), str(reviewerID_list[i]), str(category)))
        conn.commit()
        conn.close()
        return "ok", 200

#구매자 로그인
class buyersignin(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id')
        parser.add_argument('pw')
        args = parser.parse_args()

        conn = mysql.connect()
        cur = conn.cursor()
        sql = "SELECT * FROM buyer_member WHERE id=%s"
        cur.execute(sql, args['id'])

        try:
            result = cur.fetchone()
            if result['pw'] == args['pw']:
                session['idx'] = result['idx']
                session['name'] = result['name']
                session['id'] = result['id']
                data = dict(zip(('msg', 'status', 'session_info'), ('ok', 200, json.dumps(dict(zip(('idx', 'name', 'id'),(session['idx'], session['name'], session['id'])))))))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 200)
                res.headers['Content-Type'] = 'application/json'
                return res
            else:
                data = dict(zip(('msg', 'status'), ('비밀번호 틀림', 401)))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 401)
                res.headers['Content-Type'] = 'application/json'
                return res
        except Exception as e:
            data = dict(zip(('msg', 'status'), (e, 401)))
            json_data = json.dumps(data, ensure_ascii=False)
            res = make_response(json_data, 401)
            res.headers['Content-Type'] = 'application/json'
            return res

#구매자 로그아웃
class buyersignout(Resource):
    def post(self):
        session.clear()
        return 'success',200

#구매자 장바구니

#구매자 로그인 후 맞춤형 광고(카테고리 리턴)
class advertisement_category(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('idx')
        args = parser.parse_args()

        idx = args['idx']

        conn = mysql.connect()
        cur = conn.cursor()

        try:
            sql = "SELECT DISTINCT(category) FROM amazonrating WHERE reviewerID=(SELECT id FROM buyer_member WHERE idx=%s);"
            cur.execute(sql, idx)
            result = cur.fetchall()
            conn.close()

            if len(result) == 0:
                return 'failed', 401
            else:
                data = dict(zip(('msg', 'status', 'Categories'), ('ok', 200, result)))
                json_data = json.dumps(data, ensure_ascii=False)
                res = make_response(json_data, 200)
                return res
        except:
            return 'failed', 400

#구매자 카테고리별 상품정보
class productlist(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('category')
            args = parser.parse_args()

            category = args['category']

            conn = mysql.connect()
            cur = conn.cursor()

            sql = "SELECT DISTINCT(asin) FROM amazonrating WHERE category=%s"
            cur.execute(sql, category)
            result = cur.fetchall()
            conn.close()

            return result, 200
        except:
            return 'failed', 400

#구매자 평점 데이터 삽입
class buyer_rating(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('id')
            parser.add_argument('rating')
            parser.add_argument('category')
            args = parser.parse_args()

            user_name = args['id']
            rating = args['rating'].replace('[', '').replace(']','').split('}')

            del rating[-1]
            rating_list = []
            for i in rating:
                stringdata = i.replace(', {', '{')+'}'
                rating_list.append(json.loads(stringdata.replace("'", '"')))

            conn = mysql.connect()
            cur = conn.cursor()

            for i in rating_list:
                asin = i['asin']
                overall = i['overall']
                category = args['category']
                sql = "INSERT INTO amazonrating (asin, overall, reviewerID, category) VALUES (%s,%s,%s,%s)"
                cur.execute(sql, (str(asin), str(overall), str(user_name), str(category)))
                conn.commit()

            conn.close()
            return 'ok', 200
        except:
            return 'failed', 400

#svd 알고리즘으로 추천목록 도출
class svd_recommend(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('id')
            parser.add_argument('category')
            args = parser.parse_args()

            user_name = args['id']

            conn = mysql.connect()
            cur = conn.cursor()

            sql = "SELECT * FROM amazonrating WHERE category=%s"
            cur.execute(sql, args['category'])
            query = cur.fetchall()

            conn.close()

            X = dict()
            reviewerID = []
            asin = []
            overall = []
            for i in query:
                reviewerID.append(i['reviewerid'])
                asin.append(i['asin'])
                overall.append(float(i['overall']))

            X['reviewerID'] = reviewerID
            X['asin'] = asin
            X['overall'] = overall

            X = pd.DataFrame(X)

            data = X.pivot(columns='reviewerID', index='asin', values='overall')
            result = data.fillna(0)

            count = 0
            user_index = dict()
            for i in result:
                user_index[i] = count
                count += 1

            count1 = 0
            user_index_t = dict()
            for i in result:
                user_index_t[count1] = i
                count1 += 1

            u, s, v = randomized_svd(np.array(result), n_components=10, n_iter=5)
            cor_result = pd.DataFrame((s*v.T)).T.corr()

            index = user_index[user_name]
            find_top_user = cor_result[index].sort_values(ascending=False)

            dict_result = dict()
            count = 0
            for i in find_top_user.keys():
                if count < 11 and count != 0:
                    key = user_index_t[i]
                    dict_result[key] = find_top_user[i]
                    count += 1
                else:
                    count += 1

            final_result = dict_result

            test_dict = final_result

            return_array = dict()
            for i in test_dict.keys():
                return_answer = dict()
                for j in result[i].keys():
                    if result[i][j] != np.float64(0.0):
                        if result[i][j] >= 4.21:
                            return_answer[j] = result[i][j]
                return_array[i] = sorted(return_answer.items(), key=operator.itemgetter(1), reverse=True)

            frequency = {}

            for i in return_array:
                for word in return_array[i]:
                    count = frequency.get(word,0)
                    frequency[word] = count + 1

            return sorted(frequency.items(), key=operator.itemgetter(1), reverse=True), 200
        except:
            return 'failed', 400

class test2(Resource):
    def post(self):
        # down load the models from s3
        ## setting s3's model path
        bucket='sagemaker-test02'
        data_key = 'model'
        data_location = 's3://{}/{}'.format(bucket, data_key)

        # df = pd.read_csv(obj['Body'] ,header = None )
        s3 = boto3.resource('s3', aws_access_key_id=application.config['AWS_ACCESS_KEY_ID'], aws_secret_access_key=application.config['AWS_SECRET_ACCESS_KEY'])
        with BytesIO() as data:
            s3.Bucket(bucket).download_fileobj("model/svm_model_tdmvector.sav", data)
            data.seek(0)    # move back to the beginning after writing
            loaded_tdm_model = joblib.load(data)
        with BytesIO() as data:
            s3.Bucket(bucket).download_fileobj("model/svm_model_tfidf_transformer.sav", data)
            data.seek(0)    # move back to the beginning after writing
            loaded_tfidf_model = joblib.load(data)
        with BytesIO() as data:
            s3.Bucket(bucket).download_fileobj("model/svm_model_test.sav", data)
            data.seek(0)    # move back to the beginning after writing
            loaded_svm_model = joblib.load(data)
            # make sample test
            X_test = []
            X_test.append('블랙 롱패딩 아디다스 패딩 구스다운 나이키')
            y_test = 6

            # test

            X_test_tdm = loaded_tdm_model.transform(X_test)### test (transform)
            # print(X_train_tdm.shape)
            ## (2) TF_IDF

            tfidfv_test = loaded_tfidf_model.transform(X_test_tdm) #### test (transform)


            # get percentage

            target=0
            scores = loaded_svm_model.decision_function(tfidfv_test[target])
            print('input :',X_test[target])
            print('predicted value :',loaded_svm_model.predict(tfidfv_test[target]))
            print('real value :',y_test)
            print('percentage :', softmax(scores).max())

            # print(df_1[df_1['bcateid'] == loaded_model.predict(tfidfv_test[target])[0]].head())
            # # print(softmax(scores))
            # softmax(scores).sum()

        return 'ok', 200

##URL Router 맵핑 (Rest URL)

##엔드유저
#로그인
api.add_resource(signin, '/user/sign/in')
#로그아웃
api.add_resource(signout, '/user/sign/out')
#상품 등록 요청
api.add_resource(UploadImage, '/user/reqproduct')
#카테고리 수정 요청
api.add_resource(modify, '/user/modify')
#최종 등록 요청
api.add_resource(register, '/user/register')
#회원가입
api.add_resource(signup, '/user/sign/up')
#회원가입 시 중복 체크
api.add_resource(duplicatecheck, '/user/duplicatecheck')
#등록 요청 상품 조회
api.add_resource(check_n, '/user/check_n')
#등록 완료 상품 조회
api.add_resource(check_y, '/user/check_y')

#구매자
#로그인
api.add_resource(buyersignin, '/buyer/sign/in')
#로그아웃
api.add_resource(buyersignout, '/buyer/sign/out')
#상품 추천
api.add_resource(svd_recommend, '/buyer/svdrecommend')
#맞춤형광고
api.add_resource(advertisement_category, '/buyer/advertisement')
#상품 리스트
api.add_resource(productlist, '/buyer/productlist')
#리뷰
api.add_resource(buyer_rating, '/buyer/rating')

##관리자
#로그인
api.add_resource(admin_signin, '/admin/sign/in')
#로그아웃
api.add_resource(admin_signout, '/admin/sign/out')
#등록 대기 상품 조회
api.add_resource(waitproduct, '/admin/waitproduct')
#분류 정확도 일정 임계치 이상 상품 조회
api.add_resource(accurateproduct, '/admin/accurateproduct')
#분류 정확도 일정 임계치 미만 or 판매자가 수정 요청한 상품 조회
api.add_resource(inaccurateproduct, '/admin/inaccurateproduct')

api.add_resource(test2, '/qqq')
api.add_resource(temp, '/temp')

#서버 실행
if __name__ == '__main__':
    application.run(debug=True)

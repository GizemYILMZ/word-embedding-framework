import configparser
from Analysis import Analysis
import Vectorizer

Config= configparser.ConfigParser()

def ConfigSectionMap(section):
    dic={}
    options = Config.options(section)
    for option in options:
        dic[option] = Config.get(section,option)
        if dic[option] == -1:
            print("skip")
    return dic

test_file="input.ini"
Config.read(test_file)

D2V_analysis = Analysis()
create_input_path_d2v= ConfigSectionMap("create_Doc2vec_input")['create_input_path_d2v']
label= ConfigSectionMap("create_Doc2vec_input")['label']
create_input = ConfigSectionMap("create_Doc2vec_input")['create_input']

if create_input == "True":
    D2V_analysis.create_Doc2vec_input(create_input_path_d2v,label)


create_model_path_d2v = ConfigSectionMap("create_doc2vec_model")['create_model_path_d2v']
model_name_train = ConfigSectionMap("create_doc2vec_model")['model_name_train']
size = ConfigSectionMap("create_doc2vec_model")['size']
min_count = ConfigSectionMap("create_doc2vec_model")['min_count']
dm = ConfigSectionMap("create_doc2vec_model")['dm']
window = ConfigSectionMap("create_doc2vec_model")['window']
iter = ConfigSectionMap("create_doc2vec_model")['iter']
create_model = ConfigSectionMap("create_doc2vec_model")['create_model']

if create_model == "True":
    D2V_analysis.create_doc2vec_model(create_model_path_d2v,model_name_train,int(size),int(min_count),int(dm),int(window),int(iter))   #create_doc2vec_model(self,path,model_name,size,min_count)

print("Input file: ", test_file)
train_path_d2v= ConfigSectionMap("test_doc2vec")['train_path_d2v']
test_path_d2v = ConfigSectionMap("test_doc2vec")['test_path_d2v']
input_model = ConfigSectionMap("test_doc2vec")['input_model']
percentage = ConfigSectionMap("test_doc2vec")['percentage']
n =  ConfigSectionMap("test_doc2vec")['n']
size =  ConfigSectionMap("test_doc2vec")['size']
output_file_name=test_file
D2V_analysis.test_doc2vec(train_path_d2v, test_path_d2v, input_model, float(percentage), int(n), int(size),output_file_name)

test_path_tfidf = ConfigSectionMap("test_tfidf")['test_path_tfidf']
train_path_tfidf = ConfigSectionMap("test_tfidf")['train_path_tfidf']
use_idf = ConfigSectionMap("test_tfidf")['test_path_tfidf']
Tf_analysis = Analysis()
Tf_analysis.test_tfidf(train_path_tfidf,test_path_tfidf,float(percentage), int(n),output_file_name,use_idf)

# Doc2vec_vectorizer = Vectorizer.Doc2vecVectorizer()
# input_file = ConfigSectionMap("predict_class")['input_file']
# input_model= ConfigSectionMap("predict_class")['input_model']
# classes = Doc2vec_vectorizer.predict_class(input_model,input_file)
# print(classes)


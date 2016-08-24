from bottle import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.models import Sequential
from numpy import arange, sin, pi, random,cos
import math
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD,RMSprop,Adagrad,Adam,Adadelta,Adamax
from keras.models import model_from_json
#import numpy as np
#import matplotlib.pyplot as p
import matplotlib.dates as mdates


np.random.seed(1234)
from scipy import signal
result_std_test=0       #for jiska actual dekhta uska 
result_mean_test=0
result_std_testf=0      #for futureew 
result_mean_testf=0
y_test_mean=0
y_test_std=0
X_test_std=0
X_test_mean=0
y_train_mean=0
y_train_std=0
X_train_std=0
X_train_mean=0
y_orignal_test=[]
future_size_steps=0
total_loop_count=0
orignalytrainlen=0
count=0
max_value=0


mainpredicted=[]
window_length = 0
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
epochs = 1
batch_size = 50
csvip='ew1.csv'
fname='dd'



def z_normx(result):
    #print "result",result
    result = result - X_train_mean
    #print "result",result
    result = result / X_train_std
    #print "result",result
    return result, X_train_mean,X_train_std

def z_normy(result):
    #print "result",result
    result = result - y_train_mean
    #print "result",result
    result = result / y_train_std
    #print "result",result
    return result, y_train_mean,y_train_std

def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    
    return result, result_mean,result_std



def preprocessing_input(train_start, train_end,
                          test_start, test_end):
    data = np.loadtxt(fname,dtype=float,delimiter=',',skiprows=0,usecols=(2,))
    
    

    #                                       train data
    #print "Creating train data..."
    #print "********************************* DATA ************************************* \n \n "    
    result = []
    for index in range(train_start, train_end - window_length+2):
        result.append(data[index: index + window_length])
        #print "index",index,"resu_idx",result[index]
    result = np.array(result)  # shape (samples, window_length)
    global y_train_mean
    global y_train_std
    global X_train_std
    global X_train_mean

    train = result[train_start:train_end, :]
    #print "result",result
 
    X_train = train[:, :-1]

    y_train = train[:, -1]
    #print "X_train", X_train
    #print "y_train", y_train
    
    X_train,X_train_mean,X_train_std=z_norm(X_train)
    #print X_train_mean,X_train_std
    y_train,y_train_mean,y_train_std=z_norm(y_train)
    #print y_train_mean,y_train_std
    #f=input()
    
    result=[]
    result.append(data[train_end - window_length + 2 : train_end + 2])
    print "result X_test",result
    result = np.array(result)
    X_test = result[:, :-1]
    
    result = []
    for index in range(test_start- window_length+2, test_end - window_length+2):
        result.append(data[index: index + window_length])
	#print data[index: index + window_length]
    result = np.array(result)
    
    y_test = result[:, -1]
    print len(y_test)
    global X_test_std
    global X_test_mean
    global y_test_mean
    global y_test_std

    X_test,X_test_mean,X_test_std=z_normx(X_test)
    y_test,y_test_mean,y_test_std=z_normy(y_test)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print "afert reshape",X_test
    #f=input()
    
    return X_train, y_train, X_test, y_test
####################################################################################################33


def processing(X_train, y_train):
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 512, 'hidden3': 256, 'output': 1}

    model.add(LSTM(
            input_length=window_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))




    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

 
    model.compile(loss="mse", optimizer="adam")


    print("Training...")
    model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=epochs, validation_split=0.2)
    print("Predicting...")
    print("Saving the neural net.......")
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5',overwrite=True)
    print("Done....Saving")    
     
    return model



def run_network(model=None, data=None):
    j=0
    global future_size_steps
    global total_loop_count
    #future_size_steps=int(raw_input("enter future step size"))
    data22 = np.loadtxt(fname,dtype=float,delimiter=',',skiprows=0,usecols=(2,))
    data22len= len(data22)
    #input()
    train_size=int(1*data22len)
    global max_value
    with open(fname) as f:
        next(f)
        max_value=max(row[2] for row in csv.reader(f))
    
    
    
    total_loop_count=int((data22len-train_size-2)/future_size_steps)
	
    		#8 is (1439-1282)/20
    
    	#if data is None:
    print 'Loading data... '
    	    
    X_train, y_train, X_test, y_test = preprocessing_input(0,train_size-1,train_size-20,data22len-1)
    	    
    	     
    	#else:
    	 #   X_train, y_train, X_test, y_test = data
	
    print '\nData Loaded. Compiling...\n'
	
    	#if model is None:
    model = processing(X_train, y_train)
	
    	#display(X_train, y_train, X_test, y_test,model)
    
        




def testf():
	print "hello"



@route("/")
def root():
	return template("hello")
	
@get("/model")
def dmodel():
    return template("index")

@get("/optimizer")
def doptimizer():
    return template("index2")


@post("/output")
def pr1d():
	global csvip
        global fname
	global window_length
	global future_size_steps
	print (csvip)
	window_length=request.forms.get("wl")
	csvip=request.files.get("csvip")
	content=csvip.file.read()
	fname=csvip.filename
    	save_path="/home/abhi/Druva/final_22june/finaltraincode"
	#save_path="/home/prasad/Druva/22"
	file_path="{path}/{file}".format(path=save_path,file=csvip.filename)
        with open(file_path,'w') as open_file:
            open_file.write(content)
	#csvip.save(file_path)
	print(fname)
	#print(content)
        #print type(csvip) 
        #print type(fname)
        fname=str(fname)
        #print type(fname)
	#input()
    	future_size_steps=request.forms.get("fp")
    	future_size_steps= int(future_size_steps)
	window_length=int(window_length)
	#print (csvip)
	#input()
        pictname='graph.png'
	os.system("lsblk")
	run_network()
	return template("jjj",picture=pictname)

@route('/home/abhi/Druva/final_22june/finaltraincode/<picture>')
def serve_pictures(picture):
    return static_file(picture, root='/home/abhi/Druva/final_22june/finaltraincode/')

'''
@route("/")
def root():
	return template("t_welcome")
	
@get("/signup")
def getSignup():
	return template("t_signup")
	
@post("/signup")
def postSignup():
	uname=request.forms.get("user")
	password=request.forms.get("password")
	
	connection=pymongo.Connection(connection_string,safe=True)
	db=connection.name
	names=db.names
	
	nameMap={"uname":uname,"password":password}
	
	names.insert(nameMap)
	
	return "Sign up Successful!!"
	
@get("/signin")
def getSignin():
	return template("t_signin")
	
@post("/signin")
def postSignin():
	uname=request.forms.get("user")
	password=request.forms.get("password")
	
	connection=pymongo.Connection(connection_string,safe=True)
	db=connection.name
	names=db.names
	
	unameList=names.find()
	
	for n in unameList:
		if(n["uname"]==uname):
			return redirect("/newsletter")
		
	return redirect("/signup")
	
@get("/signout")
def signout():
	return redirect("/")

@get("/newsletter")
def blog():
	connection=pymongo.Connection(connection_string,safe=True)
	db=connection.name
	entries=db.entries

	entryList=entries.find()
	
	fs = gridfs.GridFS(db)
	images=db.fs.files
	imageList=images.find()
	imgname=[]
	for n in imageList:
		imgname.append(n["filename"])
	
	return template("t_newsletter",entryList=entryList,imgname=imgname)

@get("/newarticle")
def getNewPost():
	return template("t_newarticle")

@post("/newarticle")
def postNewPost():
	connection=pymongo.Connection(connection_string,safe=True)
	db=connection.name
	entries=db.entries

	title=request.forms.get("title")
	body=request.forms.get("body")
	newEntry={"title":title,"body":body}
	entries.insert(newEntry)
	
	fs = gridfs.GridFS(db)

	data=request.files.get('data')
	img_content=data.file.read()
	fname=data.filename

	fs.put(img_content,filename=fname)

	return redirect("/newsletter")

@route('/static/img/gridfs/<filename>')
def gridfs_img(filename):
	connection = pymongo.Connection(connection_string,safe=True)
	db = connection.name
	fs = gridfs.GridFS(db)
	thing = fs.get_last_version(filename=filename)
	response.content_type = 'image/jpeg'
	return thing'''

run(host='localhost',port=8039,debug=True)


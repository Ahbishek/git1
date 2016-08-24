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
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')    
     
    return model


def display(X_train, y_train, X_test, y_test,model):
    try:
	global count
	count=count+1
	print count
	global mainpredicted
	for inter in range(0,future_size_steps):
    		#print type(X_test)
		#print "just before model.predict  X_test ",X_test
        	predicted0 = model.predict(X_test)  # jyache  y test tyac ah corresponinig predict
        	#print "X_test_mean",X_test_mean
        	#predicted0=(predicted0 * X_test_std) + X_test_mean
        	predicted0=(predicted0 * y_test_std) + y_test_mean
		predicted1=predicted0
		result = np.array(predicted1)
		#print result
		mainpredicted.append(result[0][0])
        	#print predicted0,"################# YOUR PRESICTED O/PPPPPPPPPPPPPP##################################"
        	#confusion......................... above 2 stats
        	predn=predicted0
        	Xn=X_test * X_test_std + X_test_mean
        	#print "Xn",Xn
        	#print "predn",predn
        	vg=np.delete(Xn,0)
        	#print "vg",vg
        	bh=np.append(vg,predn)
        	#print "bh",bh
        	bh = np.array(bh)  # shape (samples, window_length)
	
        	bh = np.reshape(bh, (bh.size,))
		#then passed new_bh_x ... to predict  output    7.03  aya  
		bh,bh_test_mean,bh_test_std=z_normx(bh)
		#print bh
	        #print type(bh)
		rel=[]
		rel.append(bh[:])
		#print "rel should be 2d array",rel
		rel = np.array(rel)
	    	new_bh_x = rel[:, :]
		new_bh_x = np.reshape(new_bh_x, (new_bh_x.shape[0], new_bh_x.shape[1], 1))
	
		#print new_bh_x
		X_test=new_bh_x
        	
        	predicted0 = np.reshape(predicted0, (predicted0.size,))
    except KeyboardInterrupt:
            print("prediction exception")
            print 'Training duration (s) : ', time.time() - global_start_time
            return model, y_test, 0
    try:
	#data = np.loadtxt(r'ew1c71_22.csv',dtype=float,delimiter=',',skiprows=0,usecols=(2,))
	if count==1:
		global orignalytrainlen
		global y_orignal_test
		orignalytrainlen=len(y_train)
		y_orignal_test=y_test
	if count==total_loop_count:
		global orignalytrainlen
        	plt.figure(figsize=(20,10))
		
        	#plt.subplot(311)
		#print "y_above  ytest"
        	#plt.title("Actual Test Signal vs prediction  ")
        	#y_test=(y_test * result_std_test)+result_mean_test
		#print "_______________________  plot ke yaha ytest"
		#print y_test
        	y_test=(y_test * y_test_std) + y_test_mean
		y_train=(y_train * y_train_std) + y_train_mean
		y_all=np.concatenate((y_train,y_test))
		
		#print "y_all"
		#print y_all                                                   	               
		days = np.loadtxt("page-impressions.csv", unpack=True,usecols=(0,),delimiter=',',converters={ 0: mdates.strpdate2num('%Y-%m-%d %H:%M')})
		print days
		print y_all
		days=days[window_length:]
		print len(days)
		print len(y_all)
        	plt.plot_date(x=days,y=y_all, fmt='r')
		a=np.zeros(orignalytrainlen)
		predgraph=np.concatenate((a,mainpredicted))
		
		
		print len(predgraph)
			
        	plt.plot_date(x=days[:len(predgraph)],y=predgraph, fmt='g')



        	#plt.subplot(312)
        	#plt.title("Squared Error")
		#mse=1
        	#plt.plot(mse, 'r')
		y_orignal_test=(y_orignal_test * y_test_std) + y_test_mean
        	sum=0
		print "y_orignal length ",len(y_orignal_test)
		print "main predicted length ",len(mainpredicted)
        	for i in range(len(mainpredicted)):
        	    res=abs(y_orignal_test[i]-mainpredicted[i])
		    print y_orignal_test[i],"    ----------------   ",mainpredicted[i]

		    
        	    print "sub ",res
        	    res=res*res
        	    sum=sum+res
        	print "SUM",sum
        	print "len" ,len(mainpredicted)
        	sum=sum/len(mainpredicted)
		#print "sum",sum
        	sum=math.sqrt(sum)
        	print "RMSE  :- ",sum
		sum=str(sum)
                a="RMSE : "+sum
                a=str(a)
                mean2=X_train_mean      
                mean2=str(mean2)
                global max_value
            	plt.title("Actual Test Signal vs prediction  :"+a+" Max value :"+max_value+" Mean value :"+mean2)
        	plt.savefig('graph.png')

        	#plt.show()
    except Exception as e:
        print("plotting exception")
        print str(e)
  

    




def run_network(model=None, data=None):
    j=0
    global future_size_steps
    global total_loop_count
    #future_size_steps=int(raw_input("enter future step size"))
    data22 = np.loadtxt(fname,dtype=float,delimiter=',',skiprows=0,usecols=(2,))
    data22len= len(data22)
    #input()
    train_size=int(0.9*data22len)
    global max_value
    with open(fname) as f:
        next(f)
        max_value=max(row[2] for row in csv.reader(f))
    
    
    
    total_loop_count=int((data22len-train_size-2)/future_size_steps)
	
    for i in range(0,total_loop_count):		#8 is (1439-1282)/20
    
    	#if data is None:
    	print 'Loading data... '
    	    
    	X_train, y_train, X_test, y_test = preprocessing_input(0,train_size+j,train_size+1+j,data22len-1)
    	    
    	     
    	#else:
    	 #   X_train, y_train, X_test, y_test = data
	
    	print '\nData Loaded. Compiling...\n'
	
    	#if model is None:
    	model = processing(X_train, y_train)
	
    	display(X_train, y_train, X_test, y_test,model)
    
    	j=j+future_size_steps

        




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
	save_path="/home/prasad/Druva/22/finaltrial/finaltraincode"
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
        pictname='ptrf.png'
	os.system("lsblk")
	run_network()
	return template("jjj",picture=pictname)

@route('/home/prasad/Druva/22/finaltrial/finaltraincode/<picture>')
def serve_pictures(picture):
    return static_file(picture, root='/home/prasad/Druva/22/finaltrial/finaltraincode/')

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


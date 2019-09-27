"""The template of the main script of the machine learning process
"""
import pickle
import numpy as np

with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-41-26.pickle" , "rb") as f1:
    data_list1 = pickle.load(f1)
with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-41-04.pickle" , "rb") as f2:
    data_list2 = pickle.load(f2)
with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-40-29.pickle" , "rb") as f3:
    data_list3 = pickle.load(f3)
with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-40-00.pickle" , "rb") as f4:
    data_list4 = pickle.load(f4)
with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-38-31.pickle" , "rb") as f5:
    data_list5 = pickle.load(f5)
with open("C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\log\\2019-09-27_22-39-03.pickle" , "rb") as f6:
    data_list6 = pickle.load(f6)

Frame = []
Status = []
BallPosition = []
PlatformPosition = []
Bricks = []
LRUP = []
last_ball_x = 0
last_ball_y = 0
for k in range(0,8):
    for i in range(0 , len(data_list1)):
        Frame.append(data_list1[i].frame)
        Status.append(data_list1[i].status)
        BallPosition.append(data_list1[i].ball)
        PlatformPosition.append(data_list1[i].platform)
        Bricks.append(data_list1[i].bricks)
        if(i>=-1):
            if(last_ball_x - data_list1[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list1[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list1[i].ball[0]
        last_ball_y = data_list1[i].ball[1]
    for i in range(0 , len(data_list2)):
        Frame.append(data_list2[i].frame)
        Status.append(data_list2[i].status)
        BallPosition.append(data_list2[i].ball)
        PlatformPosition.append(data_list2[i].platform)
        Bricks.append(data_list2[i].bricks)
        if(i>=-1):
            if(last_ball_x - data_list2[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list2[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list2[i].ball[0]
        last_ball_y = data_list2[i].ball[1]
    for i in range(0 , len(data_list3)):
        Frame.append(data_list3[i].frame)
        Status.append(data_list3[i].status)
        BallPosition.append(data_list3[i].ball)
        PlatformPosition.append(data_list3[i].platform)
        Bricks.append(data_list3[i].bricks)
        if(i>=-1):
            if(last_ball_x - data_list3[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list3[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list3[i].ball[0]
        last_ball_y = data_list3[i].ball[1]
    for i in range(0 , len(data_list4)):
        Frame.append(data_list4[i].frame)
        Status.append(data_list4[i].status)
        BallPosition.append(data_list4[i].ball)
        PlatformPosition.append(data_list4[i].platform)
        Bricks.append(data_list4[i].bricks)
        if(i>=-1):
            if(last_ball_x - data_list4[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list4[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list4[i].ball[0]
        last_ball_y = data_list4[i].ball[1]
    for i in range(0 , len(data_list5)):
        Frame.append(data_list5[i].frame)
        Status.append(data_list5[i].status)
        BallPosition.append(data_list5[i].ball)
        PlatformPosition.append(data_list5[i].platform)
        Bricks.append(data_list5[i].bricks)  
        if(i>=-1):
            if(last_ball_x - data_list5[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list5[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list5[i].ball[0]
        last_ball_y = data_list5[i].ball[1]
    for i in range(0 , len(data_list6)):
        Frame.append(data_list6[i].frame)
        Status.append(data_list6[i].status)
        BallPosition.append(data_list6[i].ball)
        PlatformPosition.append(data_list6[i].platform)
        Bricks.append(data_list6[i].bricks)
        if(i>=-1):
            if(last_ball_x - data_list6[i].ball[0] > 0):
                LR = 1
            else:
                LR = 0
            if(last_ball_y - data_list6[i].ball[1] > 0):
                UP = 0
            else:
                UP = 1
            LRUP.append(np.array((LR,UP)))
        last_ball_x = data_list6[i].ball[0]
        last_ball_y = data_list6[i].ball[1]
    
PlatX = np.array(PlatformPosition) [:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instrust = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

Ballarray = np.array(BallPosition[:-1])
LRUP = np.array((LRUP[:-1]))
#LRDU = np.array(LRarray,UDarray)

#print(Ballarray)
#print(LRUP)
#print(PlatX[0:-1,0])
#LeftRight = np.array(LeftRight)
x = np.hstack((Ballarray,LRUP,PlatX[0:-1,0][:,np.newaxis]))



y = instrust 
np.set_printoptions(threshold=np.inf)
#print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 41)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler,y_test)
print(acc_knn_bef_scaler)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(x_train)
#x_train_stdorm = scaler.transform(x_test)
#yknn_aft_scaler = svm.predict(x_test_stdnorm)
#acc_knn_aft_scaler = accuracy+score(yknn_afe_scaler,y_test)

filename = "C:\\Users\\fghj8\\MLGame-master\\games\\arkanoid\\knn_example.sav"
pickle.dump(knn,open(filename,"wb"))
    
            


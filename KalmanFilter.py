import math
import numpy as np
import matplotlib.pyplot as plt

def Roll(A, Y):
    RollMatrix = np.array([[1, 0, 0]
                        , [0, math.cos(Y * math.pi / 180), -math.sin(Y * math.pi / 180)]
                        , [0, math.sin(Y * math.pi / 180), math.cos(Y * math.pi / 180)]])
    return RollMatrix.dot(A)

def Pitch(A, B):
    PitchMatrix = np.array([[math.cos(B * math.pi / 180), 0, math.sin(B * math.pi / 180)]
                        , [math.cos(B * math.pi / 180), 1, -math.sin(B * math.pi / 180)]
                        , [-math.sin(B * math.pi / 180), 0, math.cos(B * math.pi / 180)]])
    return PitchMatrix.dot(A)

def Yaw(A, X):
    YawMatrix = np.array([[math.cos(X * math.pi / 180), -math.sin(X * math.pi / 180), 0]
                        , [math.sin(X * math.pi / 180), math.cos(X * math.pi / 180), 0]
                        , [0, 0, 1]])
    return YawMatrix.dot(A)

def Prediction(x, v, t, a):
    A = np.array([[1, t], [0, 1]])
    X = np.array([[x], [v]])
    B = np.array([[0.5 * t ** 2], [t]])
    X_prime = A.dot(X) + B.dot(a)
    return X_prime

def HitPointEstimation(x, v, a, dt, vbullet):
    t = x / vbullet  # approx time of flight y=x
    dx = np.zeros(len(x))
    dx[1:] = np.diff(x) / dt
    ddx = np.zeros(len(x))
    ddx[1:] = np.diff(dx) / dt
    hitpoint = x + v * t + 0.5 * a * t * t
    return hitpoint

"""
def Measure(FileName, DataLength, F, n):
    Workbook = xlrd.open_workbook(FileName)
    Worksheet = Workbook.sheet_by_index(0)
    x_observations = []
    for i in range(DataLength):
        cellV = float(Worksheet.cell_value(F * int(i / F), 1))
        x_observations.append(cellV)

    x = []
    for i in range(int(DataLength / F)):
        cellV = float(Worksheet.cell_value(F * i, n))
        x.append(cellV)

    vx_observations = np.zeros(DataLength)
    vx = np.zeros(int(DataLength / F))
    vx[1:] = np.diff(x)
    for i in range(F, DataLength - 1):
        vx_observations[i] = vx[int(i / F)]

    ax_observations = np.zeros(DataLength)
    ax = np.zeros(int(DataLength / F))
    ax[1:] = np.diff(vx)
    for i in range(F, DataLength - 1):
        ax_observations[i] = ax[int(i / F)]

    Data = np.c_[x_observations, vx_observations, ax_observations]
    return Data


def ParseFile(FileName, DataLength):
    Workbook = xlrd.open_workbook(FileName)
    Worksheet = Workbook.sheet_by_index(0)

    timedata = []
    for i in range(DataLength):
        timedata.append(Worksheet.cell_value(i, 0))

    MertKalman = []
    for i in range(DataLength):
        if (Worksheet.cell_value(i, 4)) == '':
            MertKalman.append((Worksheet.cell_value(i, 1)))
        else:
            MertKalman.append((Worksheet.cell_value(i, 4)))
"""

def CreateDataX(DataLength, F):
    x = []
    for i in range(int(DataLength)):
        cellV = 100 + 10 * int(i / F)
        if (cellV >= 250.0):
            x.append(cellV - 100.0)
        elif (cellV >= 200.0):
            x.append(400.0 - cellV)
        else:
            x.append(cellV)
    return x

def CreateData(DataLength, F, x=100, v=10, a=0, Noise=0.):
    t = int(DataLength / F)
    X = np.zeros(t)
    X[0] = x
    V = []
    """
    for i in range (t):
      V.append 
      if(i>=t/2):
        V.append(4*v)
      elif(i>=t/4):
        V.append(2*v)
      else:
        V.append(v)
    """
    for i in range(t):
        V.append(v + a * i)

    for i in range(1, t):
        X[i] = X[i - 1] + V[i]

    x_observed = np.zeros(DataLength)
    for i in range(DataLength):
        x_observed[i] = X[int(i / F)]

    return AddNoise(x_observed, DataLength, Noise)

def AddNoise(x, DataLength, Power):
    # x_mean = np.mean(x)
    noise = np.random.normal(0, np.sqrt(x[0]) * Power, DataLength)
    return x + noise

def KalmanFilter(F, DataLength, Alfa=0.3, Q_Value=0.1, R_Value=10., NoisePow=0., Estimation=1., Figure=0, ReturnType=0,
                 KT=1):
    # DeltaTime
    T = 1 / F

    # Observations
    x_actual = CreateData(DataLength, F, x=100, v=10, a=1, Noise=NoisePow)
    x_observations = CreateData(DataLength, F, x=100, v=10, a=1, Noise=NoisePow)
    x = np.zeros(int(DataLength / F))
    for i in range(int(DataLength / F)):
        x[i] = x_actual[i * F]
    for i in range(DataLength):
        x_observations[i] = x[int(i / F)]

    v_observations = np.zeros(DataLength)
    v = np.zeros(int(DataLength / F))
    v[1:] = np.diff(x)
    for i in range(F, DataLength):
        v_observations[i] = v[int(i / F)]

    a_observations = np.zeros(DataLength)
    a = np.zeros(int(DataLength / F))
    a[1:] = np.diff(v)
    for i in range(F, DataLength):
        a_observations[i] = a[int(i / F)]

    Data = np.c_[x, v, a]

    # Process
    X = np.array([[Data[0][0]], [Data[0][1]], [Data[0][2]]])
    n = len(Data[0])

    # Estimation Covariance Matrix
    A = np.array([[1, T, 0.5 * T * T], [0, 1, T], [0, 0, 1]])
    I = np.identity(n)
    H = np.zeros((n, n))
    H[0][0] = 1
    # Error in Prediction
    P = np.identity(n) * 0.1
    # Noise in Measurement
    Q = np.identity(n) * Q_Value
    # Error in Measurement
    R = np.identity(n) * R_Value

    x_estimations = np.zeros(DataLength)
    v_estimations = np.zeros(DataLength)
    a_estimations = np.zeros(DataLength)

    if (KT == 1):
        for i in range(DataLength):
            # TIME UPDATE#
            X = A.dot(X)
            P = A.dot(P).dot(A.T) + Q
            if (i % F == 0):
                if (i < DataLength * Estimation):
                    # MEASUREMENT UPDATE#
                    S = H.dot(P).dot(H.T) + R

                    K = P.dot(H.T).dot(np.linalg.pinv(S))

                    P = (I - K.dot(H)).dot(P)

                    D = Data[int(i / F)][0] - H.dot(X)
                    X = X + K.dot(D)
                    E = Data[int(i / F)][0] - H.dot(X)

                    R = Alfa * R + (1 - Alfa) * (E.dot(E.T) + H.dot(P.T).dot(H.T))
                    Q = Alfa * Q + (1 - Alfa) * (K.dot(D).dot(D.T).dot(K.T))

            x_estimations[i] = X[0][0]
            v_estimations[i] = X[1][0]
            a_estimations[i] = X[2][0]

    if (KT == 1):
        XC = np.array([[Data[0][0]], [Data[0][1]], [Data[0][2]]])

        HC = np.identity(n)
        # Error in Prediction
        PC = np.identity(n) * 0.1
        # Noise in Measurement
        QC = np.identity(n) * 0.1
        # Error in Measurement
        RC = np.identity(n) * 10

        xc_estimations = np.zeros(DataLength)
        vc_estimations = np.zeros(DataLength)
        ac_estimations = np.zeros(DataLength)

        for i in range(DataLength):
            # TIME UPDATE#
            XC = A.dot(XC)
            PC = A.dot(PC).dot(A.T) + QC
            if (i % F == 0):
                if (i < DataLength * Estimation):
                    # MEASUREMENT UPDATE#
                    SC = HC.dot(PC).dot(HC.T) + RC
                    KC = PC.dot(HC.T).dot(np.linalg.inv(SC))
                    PC = (HC - KC.dot(HC)).dot(PC)
                    YC = HC.dot(Data[int(i / F)]).reshape(n, -1)
                    XC = XC + KC.dot(YC - HC.dot(XC))
            xc_estimations[i] = XC[0][0]
            vc_estimations[i] = XC[1][0]
            ac_estimations[i] = XC[2][0]

    # Plotting the Data

    t = np.arange(0., DataLength * T, T)
    v_error = abs(v_estimations - v_observations)
    error_mean = np.mean(v_error)
    error_var = np.var(v_error)
    hitpoint = HitPointEstimation(x_estimations, v_estimations, a_estimations, T, 500)
    hitpointC = HitPointEstimation(xc_estimations, vc_estimations, ac_estimations, T, 500)
    if (ReturnType == 0):
        print("Mean:", error_mean, "\t\tVar:", error_var)

    if (Figure == 1):
        # DETAILED FIGURE WITH x,v,a ESTIMATIONS VS OBSERVATIONS
        plt.figure()
        plt.subplot(221)
        plt.plot(t, x_observations, t, x_estimations, 'r--')
        plt.subplot(222)
        plt.plot(t, v_observations, t, v_estimations, 'r--')
        plt.subplot(223)
        plt.plot(t, a_observations, t, a_estimations, 'r--')
        plt.subplot(224)
        # FIGURE WITH x,v,a ESTIMATIONS VS OBSERVATIONS
        plt.plot(t, v_error)
        plt.show()
    elif (Figure == 2):
        # FIGURE of X ESTIMATIONS VS OBSERVATIONS
        plt.figure()
        plt.plot(t, x_observations, t, xc_estimations, 'g', t, x_estimations, 'r--')
        plt.show()
    elif (Figure == 3):
        plt.figure()
        plt.plot(t, x_observations, 'b', t, x_estimations, 'r--', t, hitpoint, 'y')
        plt.show()
    if (ReturnType == 1):
        return error_mean
    if (ReturnType == 2):
        return error_var

def Main():

    print('\nAdaptive Kalman Filter Graph with 10% noise with data x=100+10t+0.5t^2')
    KalmanFilter(F=20, DataLength=2000, Q_Value=10 ** -8, R_Value=0.1, NoisePow=0.3, Estimation=4/5, Figure=2, KT=1)

    """
    #TEST CASES#
  
    BestMean=[]
    temp=0.0
    for Noise in range(1,4):  #Noise level
      for Hz in range(1,5):  #Prediction Update Rate  
        Index=[0,0]
        Min=float(10**10)
        for Q in range(-3,6):  # process noise covariance matrix 
          for R in range(-3,6):  # measurement noise covariance matrix
            temp=KalmanFilter(F=10*Hz,DataLength=500*Hz,Q_Value=10**Q,R_Value=10**R,NoisePow=Noise*0.1,Estimation=4/5,ReturnType=2)    
            if(temp<Min):
              Min = temp
              Index = [Q,R]         
        BestMean.append([Noise*10,Hz*10,10**Index[0],10**Index[1]])  
        #print('Run ',Noise,',',Hz,' complete')
  
    print('Best results for the mean test cases with data x=100+10t+0.5t^2')
    for i in range(len(BestMean)):
      print(BestMean[i])      
  

    BestVar=[]
    for Noise in range(1,4):  #Noise level
      for Hz in range(1,5):  #Prediction Update Rate  
        Index=[0,0]
        Min=float(10**10)
        for Q in range(-3,6):  # process noise covariance matrix 
          for R in range(-3,6):  # measurement noise covariance matrix
            temp=KalmanFilter(F=10*Hz,DataLength=500*Hz,Q_Value=10**Q,R_Value=10**R,NoisePow=Noise*0.1,Estimation=4/5,ReturnType=2)     
            if(temp<Min):
              Min = temp
              Index = [Q,R]         
        BestVar.append([Noise*10,Hz*10,10**Index[0],10**Index[1]])  
        #print('Run ',Noise,',',Hz,' complete')
  
    print('Best results for the variance test cases with data x=100+10t+0.5t^2')
    for i in range(len(BestVar)):
      print(BestVar[i])    
 
    """

Main()

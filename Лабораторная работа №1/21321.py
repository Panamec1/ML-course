import numpy
import pandas
import matplotlib.pyplot as plt
import math


def wr(gr):
    a = ["Dr. Manthetten dist", "Mr. Euvchlid dist", "Tov. Chebyishev dist"]
    b = ["Kern: Uniform", "Kern: Triangular",
         "Kern: Epanechnikov", "Kern: Quartic"]
    c = ["Window: fixed neighbors", "Window: variable neighbors"]
    return a[gr[0]]+" | "+b[gr[1]]+" | "+c[gr[2]]



def divided(dividend, divisor):
    if divisor == 0:
        return 0
    return dividend / divisor


# Функция расстояния
def distance(distan, a, b):
    # 0 - manhattan
    # 1 - euclidean
    # 2 - chebyshev
    ans = 0
    if distan == 0:
        for x in range(len(a)):
            ans = ans + abs(a[x]-b[x])
        return ans
    if distan == 1:
        for x in range(len(a)):
            ans = ans + (a[x]-b[x])*(a[x]-b[x])
        ans = ans ** (1/2)
        return ans
    for x in range(len(a)):
        ans = max(ans, abs(a[x]-b[x]))
    return ans



# Функция ядра
def kern(kernl, x):
    if x>=1:
        return 0
    # 0 - uniform
    # 1 - triangular
    # 2 - epanechnikov
    # 3 - quartic
    if kernl == 0:
        return 1/2
    if kernl == 1:
        return 1-abs(x)
    if kernl == 2:
        return 3/4*(1-x*x)
    return 15 / 16 * (1 - x ** 2)**2
    
    



# Ядерное сглаживание
def ydernSgl(nameDist, nameKern, window_type, h, params, vals, this_param):


    #            Sum (K(p(xi,x)/hy))*yi
    # formula = ------------------------
    #              Sum K(p(xi,x)/hy)

    objects = []
    for i in range(len(params)):
        objects.append({"param": params[i],"val": vals[i]})
    objects.sort(key=lambda some_param: distance(nameDist, this_param, some_param["param"]))

    # hy
    hy = h
    if window_type == 1:
        # Не фиксированная ширина окна
        # p(u,x(u,k+))
        hy = distance(nameDist, this_param, objects[h]["param"])
    #


    
    ki = numpy.zeros(len(vals[0]))
    kern_sum = 0
    for obj in objects:
        curr_w = 0
        p = distance(nameDist, this_param, obj["param"])

        # Элемент нижней суммы
        if p == 0 or hy != 0:
            curr_w = kern(nameKern, divided(p, hy))

        # Элементы верхней суммы
        for i in range(len(vals[0])):
            ki[i] = ki[i] + obj["val"][i] * curr_w

        # Нижняя сумма
        kern_sum = kern_sum + curr_w

    # Подсчет формулы
    for i in range(len(vals[0])):
        if kern_sum == 0:
            for obj in objects:
                ki[i] = ki[i] + obj["val"][i]
            ki[i] = ki[i] / len(params)
        else:
            ki[i] = ki[i] / kern_sum

    return ki





def norm(data):
    for clazz in data:
        # Norm
        if data[clazz].dtype.name == 'int64':
            data[clazz] = (data[clazz] - min(data[clazz])) / (max(data[clazz]) - min(data[clazz]))

        # OneHot преобразование
        if data[clazz].dtype.name == 'object':
            data = pandas.concat([data.drop(clazz, axis=1), pandas.get_dummies(data[clazz])], axis=1)
    return data


# Подсчет f-меры
def fmeg(m,k):
    #print(m)

    #       2 * Precision * Recall
    # F = -------------------------
    #       Precision + Recall


    
    allin = 0

    # Массив положительных отмеченных, как положительным
    TP = []
    classes = []
    predicted = []

    
    for i in range(k):
        TP.append(m[i][i])
        
        clas = 0
        predictions = 0
        
        for j in range(k):
            allin         = allin + m[i][j]
            
            # left-right
            clas          = clas + m[i][j]
            # up-down
            predictions   = predictions + m[j][i]
            
        classes.append(clas)
        predicted.append(predictions)


    cl = 0
    sumprec = 0
    sumrec = 0
    for i in range(k):
        
        sumrec  = sumrec  + divided(TP[i],TP[i]+predicted[i])
        cl = cl+ classes[i]
        
        
        sumprec = sumprec + divided(TP[i] , TP[i]+classes[i])
    rec = divided(sumrec, k)
    prec = divided(sumprec, k)
    

    return divided(2 * prec * rec, (prec + rec))


def leave(nameDist, nameKer, nameWind, h, params, vals):
    # validation: leave one out
    k = len(vals.columns)
    matrix = [[0 for j in range(k)] for i in range(k)]
    for i in range(len(params)):
        
        curr_val = ydernSgl(nameDist,
                            nameKer,
                            nameWind, h,
                            params.drop(i, axis=0).values.tolist(),
                            vals.drop(i, axis=0).values.tolist(),
                            params.iloc[i].tolist())
        #print(numpy.argmax(vals.iloc[i]),numpy.argmax(curr_val))
        matrix[numpy.argmax(vals.iloc[i])][numpy.argmax(curr_val)] += 1
    f=fmeg(matrix,k)
    return f
  

# Запускает поиск f-меры
# для окон нефискированного размера
def varWind(max_dist,metric, kernel, window, xs, ys, bF):
    b=[False]
    for h in range(1, math.ceil(len(xs) ** (1 / 2))):
        print("now size of VARIABLE window is", h)
        f = leave(metric, kernel, window, h, xs, ys)
        if f > bF:
            b = [True,metric, kernel, window, h,f]
            bF = b[5]
    return b


# Запускает поиск f-меры
# для окон фиксироованного размера
def fixWind(max_dist,metric, kernel, window, xs, ys, bF):
    b=[False]
    curr_dist = max_dist / (len(xs) ** (1 / 2))
    while curr_dist <= max_dist:
        print("now size of FIXED window is", curr_dist)
        f = leave(metric, kernel, window, curr_dist, xs, ys)
        if f > bF:
            b = [True,metric, kernel, window, curr_dist,f]
            bF = b[5]
        curr_dist += max_dist / (len(xs) ** (1 / 2))
    return b
    

def maxDist(dist_type, params):
    res = 0
    for i in range(len(params)):
        for j in range(len(params)):
            res = max(res, distance(dist_type, params[i], params[j]))

    return res


def bester():
    # metric
    # kern
    # wind
    # wind size
    # f
    bs=[0,0,0,0,0]
    for i in range(0,3):
        max_dist = maxDist(i, xs.values.tolist())
        print(max_dist)
        for j in range(0,4):
            for k in range(0,2):
                print (wr([i,j,k]))
                if k == 0:
                    gr=fixWind(max_dist, i, j, k, xs, ys, bs[4])
                
                if k == 1:
                    gr=varWind(max_dist, i, j, k, xs, ys, bs[4])
                
                if gr[0]:
                    bs[0] = gr[1]
                    bs[1] = gr[2]
                    bs[2] = gr[3]
                    bs[3] = gr[4]
                    bs[4] = gr[5]
    print(wr(bs))
    print(bs[3])
    print("Preferable F-score is", bs[4])
    return bs






# Main part
data = pandas.read_csv('data/dataset_54_vehicle.csv')
y_names = data['Class'].unique()

# Normalization + oneHot
data = norm(data)
data.to_csv('data/norm.csv')

xs = data.drop(y_names, axis=1)
ys = data[y_names]
gr = bester()



# For graphic

# metric
# kern
# wind
# wind size
# f
gr=[0,1,1,5,0.7346]


t = len(xs) ** (1 / 2)

fs = []
sizes = []
max_dist = maxDist(gr[0], xs.values.tolist())
if gr[2] == 0:
    dis = max_dist / t
    while dis <= max_dist:
        fs.append(leave(gr[0], gr[1], gr[2], dis, xs, ys))
        sizes.append(dis)
        dis = dis + max_dist / t
        
if gr[2] == 1:
    for h in range(1, math.ceil(t)):
        fs.append(leave(gr[0], gr[1], gr[2], h, xs, ys))
        sizes.append(h)
print(max(fs))
plt.plot(sizes, fs)
plt.xlabel("The size of the window")
plt.ylabel("F-мера")
plt.show()

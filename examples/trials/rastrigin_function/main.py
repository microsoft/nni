import nni
import math
import time

def Rastrigin(A, x):
    return A*len(x) + sum([xi*xi - 10*math.cos(2*math.pi*xi) for xi in x])

def Ackley(x, y):
    return -20*math.exp(-0.2*math.sqrt(0.5*(x*x+y*y))) - math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))) + math.e + 20

def main():
    '''@nni.variable(nni.uniform(-10, 10),name=x)'''
    x = 0
    '''@nni.variable(nni.uniform(-10, 10),name=y)'''
    y = 0
    ret = Ackley(x, y)
    time.sleep(60)
    '''@nni.report_final_result(ret)'''
    print('--------------')

if __name__ == '__main__':
    main()





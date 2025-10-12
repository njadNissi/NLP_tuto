import math
import matplotlib.pyplot as plt

def plot_with_temperature(name_list,value_list,temperature):
    tmp_list = [ math.pow(math.e, x/temperature) for x in value_list]
    sum_value = sum(tmp_list)
    out_list = [  x / sum_value for x in tmp_list]
    plt.bar(name_list,out_list)
    plt.show()
    pass

if __name__ == "__main__":
    name_list = ["cat","cheese","pizza","cookie","fondue","banana","baguette","cake"]
    value_list = [3, 70, 40, 65, 55 , 10, 15 ,12 ]
    plot_with_temperature(name_list, value_list, temperature=1)
    plot_with_temperature(name_list, value_list, temperature=10)
    plot_with_temperature(name_list, value_list, temperature=50)
    plot_with_temperature(name_list, value_list, temperature=100)
    plot_with_temperature(name_list, value_list, temperature=1000)
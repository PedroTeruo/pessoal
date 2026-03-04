import matplotlib.pyplot as plt
import pandas as pd
ano = [2018, 2019, 2020, 2021, 2022]
vendas = [150, 200, 250, 300, 400]
def plotar(dataframe, x_column, y_column, title):
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe[x_column], dataframe[y_column], marker='o')
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()
df = pd.DataFrame({'Ano': ano, 'Vendas': vendas})
plotar(df, 'Ano', 'Vendas', 'Vendas Anuais')
from bokeh.plotting import figure, curdoc
from bokeh.driving import linear
import random
import pandas as pd

index_num = 0
df = pd.read_csv("./sample.csv",names=["ts","value","anomaly"])
df_length = df.shape[0]
p = figure(plot_width=1000, plot_height=400)
r1 = p.line([], [], color="firebrick", line_width=2)
r2 = p.line([], [], color="navy", line_width=2)
ano = p.circle(x=[], y=[], size=20)

ds1 = r1.data_source
ds2 = r2.data_source
anos = ano.data_source


@linear()
def update(step):
    
    ds1.data['x'].append(step)
    ds1.data['y'].append(df['value'].values[step])
    if df['anomaly'].values[step] == 1:
        anos.data['x'].append(step)
        anos.data['y'].append(df['value'].values[step])
#     ds2.data['x'].append(step)
#     ds2.data['y'].append(random.randint(0,100))  
    ds1.trigger('data', ds1.data, ds1.data)
    anos.trigger('data', anos.data, anos.data)
#     ds2.trigger('data', ds2.data, ds2.data)
    
    

curdoc().add_root(p)
index_num = 0
# Add a periodic callback to be run every 500 milliseconds
curdoc().add_periodic_callback(update, 500)
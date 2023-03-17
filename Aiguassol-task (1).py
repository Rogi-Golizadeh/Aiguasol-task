#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


# In[2]:


df=pd.read_csv(r'H:\Aiguasol\DemandaReal_base.csv')


# In[3]:


f = open(r'H:\Aiguasol\DemandaReal_base.csv')
print(f.readline())


# In[4]:


for value in df.iterrows():
    print(value)


# In[5]:


value = df['value']
print(value)


# In[6]:


value = df['value']
for i in value:
    RATE = i
DURATION = 5  # Seconds

def generate_sine_wave(freq, rate, duration):
    x = np.linspace(0, duration, rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 2 hertz sine wave that lasts for 10 minutes
x, y = generate_sine_wave(2, RATE, DURATION)
plt.plot(x, y)
plt.show()


# In[8]:


# Extract y values from the appropriate column
y = df['value'].values

# Generate time data
t = np.linspace(0, 86400, len(y))

# Compute FFT
y_fft = np.fft.fft(y)

# Compute frequencies corresponding to FFT coefficients
freqs = np.fft.fftfreq(len(y_fft)) * len(y_fft)

# Plot the original signal and its FFT
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(t, y)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[1].stem(freqs, np.abs(y_fft))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
plt.show()


# In[ ]:


from flask import Flask
app = Flask(__name__)

data = df[freqs]


@app.route('/data', methods=['GET'])
def get_data():
    return {'data': data}


if __name__ == '__main__':
    app.run()


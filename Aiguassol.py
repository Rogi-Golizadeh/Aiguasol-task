#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
app = Flask(__name__)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


# Read data from Excel file
df = pd.read_csv(r'H:\Aiguasol\DemandaReal_base.csv')

# Extract y values from the appropriate column
y = df['value'].values

# Generate sample data
t = np.linspace(0, 0.5, len(y))

# Compute FFT
y_fft = np.fft.fft(y)

# Compute frequencies corresponding to FFT coefficients
freqs = np.fft.fftfreq(len(y_fft)) * len(y_fft)
print(freqs)

# Plot the original signal and its FFT
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(t, y)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[1].stem(freqs, np.abs(y_fft))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
plt.show()
@app.route('/data', methods=['GET'])
def get_data():    
    return {'freq': f'{freqs}'}
if __name__ == '__main__':
    app.run()


# In[ ]:





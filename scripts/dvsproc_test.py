"""This script tests dvsproc module.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import scipy.signal as ss
import matplotlib.pyplot as plt
from spikefuel import dvsproc

# file_path = "./data/v_Biking_g01_c03.aedat"
# file_path_1 = "./data/vot_recordings_30fps/birds1.aedat"
file_path_1 = "./data/flashing_moving_square_LCD_tests.aedat"
file_path_2 = "./data/powerspectrum_test.aedat"
# file_path = "./data/v_BaseballPitch_g01_c01.aedat"
# video_path = "./data/v_Biking_g01_c03.avi"
# video_path = "./data/v_BaseballPitch_g01_c01.avi"

(timestamps, xaddr, yaddr, pol) = dvsproc.loadaerdat(file_path_1)
print "[MESSAGE] DATA IS LOADED."
event_arr = dvsproc.cal_event_count(timestamps)
print "[MESSAGE] EVENT COUNT IS CALCULATED"
event_freq = dvsproc.cal_event_freq(event_arr, cwindow=1000)
print "[MESSAGE] EVENT FREQUENCY IS CALCULATED"
f, pxx_den = ss.periodogram(event_freq[:, 1], 1000)
print "[MESSAGE] POWERSPECTRUM DATA IS COMPUTED"

plt.figure(0)
plt.ylim([1e-4, 1e6])
plt.semilogy(f, pxx_den)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.title("Power Spectrum from Flashing Moving Square LCD Test")
plt.savefig("./data/ps_vot_data.png")

(timestamps, xaddr, yaddr, pol) = dvsproc.loadaerdat(file_path_2)
print "[MESSAGE] DATA IS LOADED."
event_arr = dvsproc.cal_event_count(timestamps)
print "[MESSAGE] EVENT COUNT IS CALCULATED"
event_freq = dvsproc.cal_event_freq(event_arr, window=1000)
print "[MESSAGE] EVENT FREQUENCY IS CALCULATED"
f, pxx_den = ss.periodogram(event_freq[:, 1], 1000)
print "[MESSAGE] POWERSPECTRUM DATA IS COMPUTED"

plt.figure(1)
plt.ylim([1e-4, 1e6])
plt.semilogy(f, pxx_den)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.title("Power Spectrum from real recording")
plt.savefig("./data/ps_real_data.png")

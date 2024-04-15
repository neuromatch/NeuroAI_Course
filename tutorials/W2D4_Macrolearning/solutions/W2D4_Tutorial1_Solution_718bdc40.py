
#define variables
A = .005
B = 0.1
phi = 0
C = 1

#define days (observe that those are not 1, ..., 365 but proxy ones to make model function neat)
days = np.arange(-26, 26 + 1/7, 1/7) #defined as fractions of a week

prices = A * days**2 + B * np.sin(np.pi * days + phi) + C

#plot relation between days and prices
with plt.xkcd():
  plt.plot(days, prices)
  plt.xlabel('Day')
  plt.ylabel('Price')
  plt.show()
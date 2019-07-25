
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def simulate_normal_demand(timesteps, I0, mu = 100, sigma = 5, showplot = False, figsize = (15, 10)):
	Inventory = {-1 : I0, -0.5 : I0}
	Demand = {}
	Shortage = {}
	Production = {}
	for i in np.arange(0, timesteps, 0.5):
		current_inv = Inventory[i-0.5]
		if i % 1 == 0:
			# deplete
			demand = mu + sigma * np.random.randn()
			if demand > current_inv:
				Shortage[i] = current_inv - demand
				new_inv = current_inv
				print('shortage at t =', i)
			else:
				Shortage[i] = 0
				new_inv = current_inv - demand
				print('demand met at t =', i)
			Demand[i] = demand
		if i % 1 == 0.5:
			# replenish
			prod = (I0 - current_inv) if current_inv < I0 else 0
			print('production of', prod, 'at time t =', i)
			new_inv = current_inv + prod
			Production[i] = prod
		Inventory[i] = new_inv
	plt.close('all')
	plt.figure(figsize = figsize)
	plt.plot(Inventory.keys(), Inventory.values(), label = 'Inventory', color = plt.cm.Reds(0.8))		
	plt.scatter(Demand.keys(), Demand.values(), label = 'Demand', color = plt.cm.Blues(0.8), s = 20, alpha = 0.8, lw = 0.2, edgecolors = 'k')
	plt.scatter(Shortage.keys(), Shortage.values(), label = 'Shortage', color = plt.cm.Greens(0.8), s = 20, alpha = 0.8, lw = 0.2, edgecolors = 'k')
	#plt.plot(Production.keys(), Production.values(), label = 'Production', color = plt.cm.jet(0.7))
	titletext = 'Timeseries for simulation of:\n'+ str(timesteps)+' steps, '
	titletext += 'constant inventory = ' + str(I0)
	titletext += ' and demand ~ N({},{})'.format(mu, sigma**2)
	plt.legend()
	plt.title(titletext)
	if showplot:
		plt.show()
	return Demand, Inventory, Shortage, Production


d, i, s, p = simulate_normal_demand(500, 75)
plt.savefig('./fig/simulations/sim01_normal_ts_3.png')

def get_new_axis(figsize = (15, 10)):
	plt.close('all')
	plt.figure(figsize = figsize)
	ax = plt.gca()
	return ax

def plot_hist(ax, data, bins = 50, logscale = False, figsize = (15, 10)):
	assert isinstance(data, dict)
	if ax is None:
		ax = get_new_axis(figsize)
	for k, v in data.items():
		if isinstance(v, dict):
			print('extracting values for key', k)
			v = list(v.values())
		assert isinstance(v, list)
		ndata = v
		if logscale:
			ndata = np.log(ndata)
		counts, edges = np.histogram(ndata, bins = bins)
		vals = [np.mean(edges[i:i+2]) for i in range(len(edges)-1)]
		ax.plot(vals, counts, label = k)
	return ax

def normal_simulation_histogram(timesteps, I0, mu = 100, sigma = 5, figsize = (15, 10), showtimeplot = False):
	d, i, s, p = simulate_normal_demand(timesteps, I0, mu, sigma, showplot = showtimeplot)
	data = dict(zip(['Demand', 'Inventory', 'Shortage', 'Production'], [d, i, s, p]))
	ax = plot_hist(None, data, figsize = figsize)
	titletext = 'Histogram for simulation of:\n'+ str(timesteps)+' steps, '
	titletext += 'constant inventory = ' + str(I0)
	titletext += ' and demand ~ N({},{})'.format(mu, sigma**2)
	plt.title(titletext)
	return ax

normal_simulation_histogram(50000, 200)
plt.legend()
plt.savefig('./fig/simulations/sim01_normal_hist_4.png')
plt.show()

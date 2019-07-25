import sys
sys.path.append('/Users/aabir/anaconda/envs/pca/simulations')

from invmgmt import *

from matplotlib.ticker import ScalarFormatter

def compare_normal_final_values(t=1000, labels=None, strategy='simple_strategy', **kwargs):
	labels = labels or ['reward_', 'missed_reward', 'extra_inventory']
	mu_vals = kwargs.pop('mu_vals', np.arange(50, 160, 25))
	sigma_vals = kwargs.pop('sigma_vals', np.arange(10, 51, 10))
	I_vals = kwargs.pop('I_vals', np.arange(50, 160, 25))
	cwd = kwargs.pop('cwd', '/Users/aabir/anaconda/envs/pca/sim_results')
	maxtime = kwargs.pop('maxtime', np.inf)
	legendfontsize = kwargs.pop('lfs', 'large')
	li = kwargs.pop('lindices', [-1, -1])
	# figure kwargs
	figsize = kwargs.pop('figsize', (15, 10))
	showplot = kwargs.pop('showplot', True)
	savefig = kwargs.pop('savefig', False)
	y_lims = kwargs.pop('y_lims', None)
	#
	hcs = kwargs.pop('hcs', [0.05, 0.2])
	for ind1 in range(len(mu_vals)):
		finalrewards, finalwaste = {}, {}
		for ind3 in range(len(sigma_vals)):
			for ind2 in range(len(I_vals)):
				sigma = sigma_vals[ind3]
				mu = mu_vals[ind1]
				I = I_vals[ind2]
				print('\tmu={}, sigma={}, I0={}'.format(mu, sigma, I))
				sim = Inventory_MGMT(I0 = I, demand_type='normal', mu=mu, sigma=sigma)
				sim.run_simulation(timesteps = t, strategy=strategy)
				max_reward, max_waste = get_largest_key_val(sim.logs['reward_'])[1], get_largest_key_val(sim.logs['extra_inventory'])[1]
				finalrewards[I] = max_reward
				finalwaste[I] = max_waste
				titletext = sim.d_gen.desc 
			fig = plt.figure(figsize = figsize)
			col = iter([plt.cm.Blues(0.8), plt.cm.Reds(0.8)] + [plt.cm.Greens(a) for a in np.arange(0.4, 1.01, 0.3)])
			x1, y1 = np.array(list(finalrewards.keys())), np.array(list(finalrewards.values()))
			x2, y2 = np.array(list(finalwaste.keys())), np.array(list(finalwaste.values()))
			#if strategy == 'ROP_strategy':
			#	x1, x2 = 5*x1, 5*x2
			plt.plot(x1, y1, c=next(col), label='sales', lw=2, alpha=0.8)
			plt.plot(x2, y2, c=next(col), label='extra inventory', lw=2, alpha=0.8)
			#plt.scatter(x1, y1, c=next(col), label='sales', s=8, alpha=0.8)
			#plt.scatter(x2, y2, c=next(col), label='extra inventory', s=8, alpha=0.8)
			#
			for hc in hcs:
				plt.plot(x1, y1-hc*y2, c=next(col), label='profit with holding cost={}'.format(format(hc, '.2f')), lw=3, alpha=0.6)
			# ticks
			if y_lims:
				plt.yticks(range(-10**5, 10**6, 10**5), [MKBformatter(i) for i in range(-10**5, 10**6, 10**5)])
				plt.ylim(y_lims)
			elif strategy=='ROP_strategy':
				plt.yticks(range(-10**7, 10**8, 10**6), [MKBformatter(i) for i in range(-10**7, 10**8, 10**6)])
				plt.ylim(-0.5*10**6, 5*10**6)
			else:
				plt.yticks(range(-10**5, 10**6, 10**5), [MKBformatter(i) for i in range(-10**5, 10**6, 10**5)])
				plt.ylim(-0.5*10**5, 1*10**6)
			# title stuff
			# titletext += 'timeseries for {} timesteps\n'
			#titletext += '\n' + r'$\mu \in {{{},...{}}}, I_0 \in {{{},...{}}}$'.format(min(mu_vals), max(mu_vals),
			#																min(I_vals), max(I_vals)) + '\n'
			plt.title(titletext)
			if strategy == 'ROP_strategy':
				plt.xlabel(r'$R_0$')
			else:
				plt.xlabel(r'$I_0$')
			#
			plt.xlim(xmin=0)
			lgnd = plt.legend(loc='upper left')
			for i in range(len(lgnd.legendHandles)):
				lgnd.legendHandles[i]._sizes = [45]
			#
			if showplot:
				plt.show()
			else:
				savechoice = savefig #or input('save timeseries?\t y/n\t') == 'y'
				if savechoice:
					folderpath = join(cwd, 'comparisons', 'final_vals', '_'.join(strategy.split(' ')))
					os.makedirs(folderpath, exist_ok=True)
					savepath = folderpath+'/normal(mu={},sigma={}).png'.format(mu, sigma)
					fig.savefig(savepath)
					print('\tsaved', savepath)


def compare_powerlaw_timeseries(t=1000, labels=None, strategy='simple_strategy', **kwargs):
	labels = labels or ['reward_', 'missed_reward', 'extra_inventory']
	alpha_vals = kwargs.pop('alpha_vals', np.arange(2, 3.2, 0.2)) 
	mu_vals = kwargs.pop('mu_vals', np.arange(50, 160, 25))
	I_vals = kwargs.pop('I_vals', np.arange(50, 160, 25))
	cwd = kwargs.pop('cwd', '/Users/aabir/anaconda/envs/pca/sim_results')
	maxtime = kwargs.pop('maxtime', np.inf)
	legendfontsize = kwargs.pop('lfs', 'large')
	li = kwargs.pop('lindices', [-1, -1])
	# figure kwargs
	figsize = kwargs.pop('figsize', (15, 10))
	showplot = kwargs.pop('showplot', True)
	savefig = kwargs.pop('savefig', False)
	#
	sim_list = {}
	for alpha in alpha_vals:
		print('running ensemble for alpha={}'.format(alpha))
		plt.close('all')
		fig, axs = plt.subplots(len(mu_vals), len(I_vals), sharex=True, sharey=True, figsize=figsize, squeeze=False)
		for ind1 in range(len(mu_vals)):
			for ind2 in range(len(I_vals)):
				mu = mu_vals[ind1]
				I = I_vals[ind2]
				#print('\tmu={}, I0={}'.format(mu, I))
				sim = Inventory_MGMT(I0 = I, demand_type='powerlaw', alpha=alpha, mu=mu)
				sim.run_simulation(timesteps = t, strategy=strategy)
				ax = axs[ind1][ind2]
				colors = iter(plt.cm.gnuplot(i/len(labels)) for i in range(len(labels)))
				for k in labels:
					single_timeseries(ax, sim.logs[k], k, col = next(colors), maxtime = maxtime)
				if ind2 == 0:
					ax.set_ylabel(r'$\mu={}$'.format(int(mu)))
				if ind1 == 0:
					ax.set_title(r'$I_0={}$'.format(int(I)))
				#ax.set_title(r'$\mu={}, I_0={}$'.format(mu, I))
				if (mu, I) == (mu_vals[li[0]], I_vals[li[1]]):
					ax.legend(fontsize = legendfontsize, loc=(-0.75, 1.1))
				ax.set_xlim(xmin=0)
				ax.set_ylim(ymin=0)
		# title stuff
		titletext = ''
		# titletext += 'timeseries for {} timesteps\n'
		titletext += 'cumulative timeseries behaviour\ndemand: powerlaw(alpha={})'.format(format(alpha, '.2f'))
		#titletext += '\n' + r'$\mu \in {{{},...{}}}, I_0 \in {{{},...{}}}$'.format(min(mu_vals), max(mu_vals),
		#																min(I_vals), max(I_vals)) + '\n'
		#
		plt.suptitle(titletext, y = 0.98, weight = 'bold')
		# layout stuff
		plt.tight_layout(rect=[0, 0.03, 1, 0.94])
		if showplot:
			plt.show()
		else:
			savechoice = savefig #or input('save timeseries?\t y/n\t') == 'y'
			if savechoice:
				folderpath = join(cwd, 'comparisons/timeseries')
				os.makedirs(folderpath, exist_ok=True)
				savepath = folderpath+'/alpha={}({}x{}).png'.format(format(alpha, '.1f'), len(mu_vals), len(I_vals))
				fig.savefig(savepath)
				print('\tsaved', savepath)

def compare_powerlaw_slopes(t=1000, labels=None, strategy='simple_strategy', **kwargs):
	labels = labels or ['reward_', 'missed_reward', 'extra_inventory']
	alpha_vals = kwargs.pop('alpha_vals', np.arange(2, 3.2, 0.2)) 
	mu_vals = kwargs.pop('mu_vals', np.arange(50, 160, 25))
	I_vals = kwargs.pop('I_vals', np.arange(50, 160, 25))
	cwd = kwargs.pop('cwd', '/Users/aabir/anaconda/envs/pca/sim_results')
	maxtime = kwargs.pop('maxtime', np.inf)
	legendfontsize = kwargs.pop('lfs', 'large')
	li = kwargs.pop('lindices', [-1, -1])
	# figure kwargs
	figsize = kwargs.pop('figsize', (15, 10))
	showplot = kwargs.pop('showplot', True)
	savefig = kwargs.pop('savefig', False)
	#
	for alpha in alpha_vals:
		mu_I = []
		data = {k : np.zeros((len(mu_vals), len(I_vals))) for k in labels}
		print('running ensemble for alpha={}'.format(alpha))
		plt.close('all')
		for ind1 in range(len(mu_vals)):
			temp = []
			for ind2 in range(len(I_vals)):
				mu = mu_vals[ind1]
				I = I_vals[ind2]
				temp += [(mu, I)]
				print('\tmu={}, I0={}'.format(mu, I))
				sim = Inventory_MGMT(I0 = I, demand_type='powerlaw', alpha=alpha, mu=mu)
				sim.run_simulation(timesteps = t, strategy=strategy)
				for k in labels:
					x, y = np.array(list(sim.logs[k].keys())), np.array(list(sim.logs[k].values()))
					vals = np.polyfit(x, y, deg = 1)
					data[k].itemset((ind1, ind2), vals[0])
			mu_I += [temp]
		print(mu_I)
		print(data)
		fig, axs = plt.subplots(1, len(data), figsize = figsize, squeeze=False)
		cms = iter([plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, 'gray'])
		axs = iter(j for i in axs for j in i)
		print('\n\nplotting slope results')
		mu_for_plot = bounds_for_pcolor(mu_vals)
		I_for_plot = bounds_for_pcolor(I_vals)
		for k in data:
			ax = next(axs)
			tplot = ax.pcolor(mu_for_plot, I_for_plot, data[k], cmap = next(cms))
			ax.set_title(k.replace('_', ' ')+' estimate')
			ax.set_ylabel(r'$\mu$')
			ax.set_xlabel(r'$I_0$')
			fig.colorbar(tplot, ax=ax, fraction=0.08, pad = 0.01)
		# title stuff
		titletext = ''
		# titletext += 'timeseries for {} timesteps\n'
		titletext += 'estimate per timestep\n with demand: powerlaw(alpha={})'.format(format(alpha, '.2f'))
		#titletext += '\n' + r'$\mu \in {{{},...{}}}, I_0 \in {{{},...{}}}$'.format(min(mu_vals), max(mu_vals),
		#																min(I_vals), max(I_vals)) + '\n'
		#
		plt.suptitle(titletext, y = 0.98, weight = 'bold')
		# layout stuff
		plt.tight_layout(rect=[0, 0.03, 1, 0.92])
		#
		if showplot:
			plt.show()
		else:
			savechoice = savefig #or input('save timeseries?\t y/n\t') == 'y'
			if savechoice:
				folderpath = join(cwd, 'comparisons/estimates')
				os.makedirs(folderpath, exist_ok=True)
				savepath = folderpath+'/slopes_w_alpha={}.png'.format(format(alpha, '.1f'), len(mu_vals), len(I_vals))
				fig.savefig(savepath)
				print('\tsaved', savepath)

def compare_powerlaw_final_values(t=1000, labels=None, strategy='simple_strategy', **kwargs):
	labels = labels or ['reward_', 'missed_reward', 'extra_inventory']
	alpha_vals = kwargs.pop('alpha_vals', np.arange(2, 3.2, 0.2)) 
	mu_vals = kwargs.pop('mu_vals', np.arange(50, 160, 25))
	I_vals = kwargs.pop('I_vals', np.arange(50, 160, 25))
	cwd = kwargs.pop('cwd', '/Users/aabir/anaconda/envs/pca/sim_results')
	maxtime = kwargs.pop('maxtime', np.inf)
	legendfontsize = kwargs.pop('lfs', 'large')
	li = kwargs.pop('lindices', [-1, -1])
	# figure kwargs
	figsize = kwargs.pop('figsize', (15, 10))
	showplot = kwargs.pop('showplot', True)
	savefig = kwargs.pop('savefig', False)
	hcs = kwargs.pop('hcs', [0.05, 0.2])
	y_lims = kwargs.pop('y_lims', None)
	#
	for alpha in alpha_vals:
		mu_I = []
		data = {k : np.zeros((len(mu_vals), len(I_vals))) for k in labels}
		print('running ensemble for alpha={}'.format(alpha))
		plt.close('all')
		finalrewards, finalwaste = {}, {}
		for ind1 in range(len(mu_vals)):
			temp = []
			for ind2 in range(len(I_vals)):
				mu = mu_vals[ind1]
				I0 = I_vals[ind2]
				temp += [(mu, I0)]
				print('\tmu={}, I0={}'.format(mu, I0))
				sim = Inventory_MGMT(I0 = I0, demand_type='powerlaw', alpha=alpha, mu=mu)
				sim.run_simulation(timesteps = t, strategy=strategy)
				max_reward, max_waste = get_largest_key_val(sim.logs['reward_'])[1], get_largest_key_val(sim.logs['extra_inventory'])[1]
				finalrewards[I0] = max_reward
				finalwaste[I0] = max_waste
				titletext = sim.d_gen.desc
			fig = plt.figure(figsize = figsize)
			col = iter([plt.cm.Blues(0.8), plt.cm.Reds(0.8)] + [plt.cm.Greens(a) for a in np.arange(0.4, 1.01, 0.3)])
			x1, y1 = np.array(list(finalrewards.keys())), np.array(list(finalrewards.values()))
			x2, y2 = np.array(list(finalwaste.keys())), np.array(list(finalwaste.values()))
			#print(x1.shape, y1.shape, x2.shape, y2.shape)
			#if strategy == 'ROP_strategy':
			#	x1, x2 = 5*x1, 5*x2
			plt.plot(x1, y1, c=next(col), label='total shipped', lw=2, alpha=0.8)
			plt.plot(x2, y2, c=next(col), label='extra inventory', lw=2, alpha=0.8)
			#plt.scatter(x1, y1, c=next(col), label='sales', s=8, alpha=0.8)
			#plt.scatter(x2, y2, c=next(col), label='extra inventory', s=8, alpha=0.8)
			for hc in hcs:
				plt.plot(x1, y1-hc*y2, c=next(col), label='profit with holding cost={}'.format(format(hc, '.2f')), lw=3, alpha=0.6)
			#
			# title stuff
			#titletext = ''
			# titletext += 'timeseries for {} timesteps\n'
			#titletext += 'performance\ndemand: powerlaw(mu={}, alpha={})'.format(mu, format(alpha, '.2f'))
			#titletext += '\n' + r'$\mu \in {{{},...{}}}, I_0 \in {{{},...{}}}$'.format(min(mu_vals), max(mu_vals),
			#																min(I_vals), max(I_vals)) + '\n'
			#
			plt.xlabel(r'$I_0$')
			if y_lims:
				plt.yticks(range(-10**5, 10**6, 10**5), [MKBformatter(i) for i in range(-10**5, 10**6, 10**5)])
				plt.ylim(y_lims)
			elif strategy=='ROP_strategy':
				plt.yticks(range(-10**7, 10**8, 10**6), [MKBformatter(i) for i in range(-10**7, 10**8, 10**6)])
				plt.ylim(-0.5*10**6, 5*10**6)
			else:
				plt.yticks(range(-10**5, 10**6, 10**5), [MKBformatter(i) for i in range(-10**5, 10**6, 10**5)])
				plt.ylim(-0.5*10**5, 1*10**6)
			#
			plt.xlim(xmin=0)
			if strategy == 'ROP_strategy':
				plt.xlabel(r'$R_0$')
			#plt.ylim(ymin=0)
			#
			plt.title(titletext)
			lgnd = plt.legend(loc='upper left')
			for i in range(len(lgnd.legendHandles)):
				lgnd.legendHandles[i]._sizes = [45]
			#
			if showplot:
				plt.show()
			else:
				savechoice = savefig #or input('save timeseries?\t y/n\t') == 'y'
				if savechoice:
					folderpath = join(cwd, 'comparisons', 'final_vals', '_'.join(strategy.split(' ')))
					os.makedirs(folderpath, exist_ok=True)
					savepath = folderpath+'/powerlaw(mu={},alpha={}).png'.format(mu, format(alpha, '.1f'))
					fig.savefig(savepath)
					print('\tsaved', savepath)

def bounds_for_pcolor(data):
	data = list(data)
	if len(data) < 2:
		return np.array([0.9*data[0], data[0]*1.1])
	delta = data[1] - data[0]
	assert all(data[i+1] - data[i] == delta for i in range(len(data) -1 ))
	return np.arange(data[0] - delta/2, data[-1] + 1.2*delta/2 , delta)


if __name__ == "__main__":
	#compare_powerlaw_timeseries(showplot=False, savefig=True, t = 70, mu_vals = range(60, 141, 20), 
	#								I_vals = range(60, 141, 20), figsize = (15, 12), lfs = 12)
	#compare_powerlaw_slopes(showplot=False, savefig=True, alpha_vals = [2.4], mu_vals = [100], 
	#									I_vals = range(10, 201, 10), figsize = (15, 8), lfs = 12)
	normal_I0_dict = {	'simple_strategy' : range(10, 501, 10),
					'bulk_order_strategy' : range(300, 701, 100),
					'ROP_strategy' : range(10, 151, 2)}
	pl_I0_dict = {	'simple_strategy' : range(10, 1001, 10),
					'bulk_order_strategy' : range(300, 701, 100),
					'ROP_strategy' : range(10, 301, 2)}
	
	for strategy in ['simple_strategy']:#, 'bulk_order_strategy']:
		continue
		compare_powerlaw_final_values(showplot=False, savefig=True, alpha_vals = [2.6], mu_vals = [100], 
										I_vals = pl_I0_dict[strategy], figsize = (10, 8), lfs = 12, strategy=strategy, y_lims=(0, 4e5))
		compare_normal_final_values(showplot=False, savefig=True, sigma_vals = [10, 30], mu_vals = [100], 
										I_vals = normal_I0_dict[strategy], figsize = (10, 8), lfs = 12, strategy=strategy, y_lims=(0, 4e5))
	for strategy in ['ROP_strategy']:
		compare_powerlaw_final_values(showplot=False, savefig=True, alpha_vals = [2.6], mu_vals = [100], 
										I_vals = pl_I0_dict[strategy], figsize = (10, 8), lfs = 12, strategy=strategy, y_lims=(0,5e5))
		compare_normal_final_values(showplot=False, savefig=True, sigma_vals = [10, 30], mu_vals = [100], 
										I_vals = normal_I0_dict[strategy], figsize = (10, 8), lfs = 12, strategy=strategy, y_lims=(0,5e5))





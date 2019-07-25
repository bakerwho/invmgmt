import sys
sys.path.append('/Users/aabir/anaconda/envs/pca/simulations')

from invmgmt import *

def run_ensemble(savefig = True, t = 1000, vals = {}, strategy='simple_strategy', **kwargs):
	mu_vals = kwargs.pop('mu_vals', range(40, 161, 20))
	I0_vals = kwargs.pop('I0_vals', range(10, 201, 10))
	#
	d_list, I0_list = [], []
	ts_y_lims = kwargs.pop('ts_y_lims', {})
	hist_x_lims = kwargs.pop('hist_x_lims', {})
	annotate = kwargs.pop('annotate', False)
	# normal
	if vals.pop('normal', False):
		sigma_vals = kwargs.pop('sigma_vals', range(10, 41, 10))
		for sigma in sigma_vals:
			for mu in mu_vals:
				for I0 in I0_vals:
					d_list += [get_normal_DemandGenerator(mu, sigma)]
					I0_list += [I0]
	# powerlaw
	if vals.pop('powerlaw', False):
		alpha_vals = kwargs.pop('alpha_vals', np.arange(2.2, 3.1, 0.4))
		for mu in mu_vals:
			for alpha in alpha_vals:
				for I0 in I0_vals:
					d_list.append(get_powerlaw_DemandGenerator(alpha, mu))
					I0_list.append(I0)
		"""
		for alpha in np.arange(1.2, 5, 0.4):
			d_list.append(get_powerlaw_DemandGenerator(alpha, 100, (0, 500)))
			I0_list.append(100)
		#
		for mu in range(50, 200, 50):
			d_list.append(get_powerlaw_DemandGenerator(2, mu, (0, 500)))
			I0_list.append(100)
		"""
	# uniform
	if vals.pop('uniform', False):
		a_vals = kwargs.pop('a_vals', range(50, 220, 50))
		b_vals = kwargs.pop('b_vals', range(100, 270, 50))
		for a in a_vals:
			for b in b_vals:
				for I0 in I0_vals:
					d_list.append(get_uniform_DemandGenerator(a, b))
					I0_list.append(I0)
	#
	to_remove = {}
	for I0, d in zip(I0_list, d_list):
		i = Inventory_MGMT(I0 = I0, d_gen = d)
		i.run_simulation(timesteps = t, strategy=strategy)
		#to_remove = {	'inventory' : [i.I0, 0], 'production' : [i.I0], 'supply' : [0, i.I0],
		#				'shortage' : [0]}
		i.plot_timeseries(showplot = False, savefig = savefig, maxtime=3000, scatterkeys = {}, y_lims=ts_y_lims)
		i.plot_histograms(showplot = False, savefig = savefig, to_remove = to_remove, x_lims=hist_x_lims, annotate=annotate)
		#if input('show logs?\ty/n\t') == 'y':
		#		print(i.logs.items())

if __name__ == "__main__":
	#run_simple_ensemble(vals={'normal': True, 'powerlaw':True, 'uniform' : False}, mu_vals = [100], I0_vals = range(10, 121, 10))
	ts_y_lims = {	'simple_strategy' : {	'demand':(0, 600), 'inventory':(0, 200), 
					'supply':(0, 200), 'production':(0, 200), 'shortage':(0, 200), 'reward_':(0, 10000)},
				'bulk_order_strategy' : {	'demand':(0, 600), 'inventory':(0, 600), 
					'supply':(0, 600), 'production':(0, 600), 'shortage':(0, 600), 'reward_':(0, 15000)},
				'ROP_strategy' : {	'demand':(0, 600), 'inventory':(0, 2000), 
					'supply':(0, 1200), 'production':(0, 1200), 'shortage':(0, 400), 'reward_':(0, 50000)}
				}
	#strategy = 'simple_strategy'
	I0_dict = {	'simple_strategy' : range(100, 40, 240),
				'bulk_order_strategy' : range(300, 701, 100),
				'ROP_strategy' : range(120, 201, 30)}
	anno = False
	for strategy in ['simple_strategy', 'ROP_strategy']:#, 'bulk_order_strategy']:
		yl = ts_y_lims[strategy]
		xl = ts_y_lims[strategy]
		I0v = I0_dict[strategy]
		run_ensemble(	t = 200, vals={'normal': True, 'powerlaw':True, 'uniform' : False}, 
						mu_vals = [100], I0_vals = I0v, sigma_vals=[10],
						strategy = strategy, ts_y_lims=yl, hist_x_lims=xl, annotate=anno)
		run_ensemble(	t = 200,vals={'normal': True, 'powerlaw':True, 'uniform' : False}, 
						mu_vals = [100], I0_vals = I0v, alpha_vals = [2.6],
						strategy = strategy, ts_y_lims=yl, hist_x_lims=xl, annotate=anno)

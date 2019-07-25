import sys
sys.path.append('/Users/aabir/anaconda/envs/pca/simulations')

from invmgmt import *

import matplotlib.ticker as mtick

def setup_results_dirs():
	cwd = '/Users/aabir/anaconda/envs/pca/'
	os.makedirs(cwd + 'sim_results/delay', exist_ok=True)

class Inventory_MGMT:
	def __init__(self, product_name='product1', d_gen=None, I0=0, **kwargs):
		self.product_name = product_name
		self.logs =	{	'inventory' 	: {-0.5 : I0},
						'demand'		: {-0.5 : 0},
						'shortage'		: {-0.5 : 0},
						'supply'		: {-0.5 : 0},
						'production'	: {-0.5 : 0},
						'backlog'		: {-0.5 : 0},
						'reward_'		: { -0.5 : 0},
						'extra_inventory'	: { -0.5 : 0},
						'missed_reward'	: { -0.5 : 0}
					}
		self.demandQ = q.PriorityQueue()
		self.productionQ = q.PriorityQueue()
		self.supplyQ = q.PriorityQueue()
		self.I0 = I0
		#
		setup_results_dirs()
		self.cwd = kwargs.pop('cwd', '/Users/aabir/anaconda/envs/pca/sim_results/delay') + '/'
		self.set_demand_generator(d_gen, kwargs=kwargs.copy())
		self.timer = 0
		self.delay_type = kwargs.pop('delay_type', 'log normal')
		self.delay_func = self.delay(self.delay_type, kwargs.copy())

	def set_demand_generator(self, d_gen=None, kwargs={}):
		assert isinstance(d_gen, DemandGenerator) or d_gen is None
		kwargs = kwargs.copy()
		if d_gen is None:
			demand_type = kwargs.pop('demand_type', None)
			if demand_type == 'normal':
				mu = kwargs.pop('mu')
				sigma = kwargs.pop('sigma')
				minmax = kwargs.pop('minmax', (0, np.inf))
				d_gen = get_normal_DemandGenerator(mu, sigma, minmax=minmax)
			elif demand_type == 'powerlaw':
				mu = kwargs.pop('mu')
				alpha = kwargs.pop('alpha')
				minmax = kwargs.pop('minmax', (0, np.inf))
				d_gen = get_powerlaw_DemandGenerator(alpha, mu, minmax=minmax)
			elif demand_type == 'uniform':
				a = kwargs.pop('a')
				b = kwargs.pop('b')
				d_gen = get_uniform_DemandGenerator(a, b)
			else:
				print(kwargs)
				raise ValueError('exiting set_demand_generator() : invalid demand')
		self.d_gen = d_gen

	def demand_step(self, d_step, t=None):
		assert callable(d_step), 'invalid demand_step function'
		t = t or self.timer
		results = d_step()
		self.update_logs(results, t)
		## add demand to demandQ

	def supply_step(self, s_step, t=None):
		## decide how much to supply, remove from inventory and add to supplyQ
		## pop from supplyQ if ready and add to rewardQ
		assert callable(s_step), 'invalid supply_step function'
		t = t or self.timer
		results = s_step()
		self.update_logs(results, t)
			
	def production_step(self, p_step, t=None):
		## decide how much to supply and add to supplyQ
		## pop from supplyQ if ready and add to inventory
		assert callable(p_step), 'invalid production_step function'
		t = t or self.timer
		results = p_step()
		self.update_logs(results, t)

	def run_simulation(self, eventsteps = 1000, steps = None, **kwargs):
		steps = self.parse_steps(steps, kwargs.copy())
		self.d_gen = self.d_gen or self.set_demand_generator(kwargs=kwargs.copy())
		assert isinstance(self.d_gen, DemandGenerator), 'Invalid d_gen in simulation'
		self.simtime = eventsteps
		self.strategy = steps['strategy']
		self.get_strategy_desc()
		print('\trunning simulation for {} eventsteps (desc: {}) with strategy {}'.format(eventsteps, 
															self.d_gen.desc.replace('$','').replace('\\', ''), 
															steps['strategy'].replace('$','').replace('\\', '')))
		for i in range(eventsteps):
			old = {k : get_largest_key_val(self.logs[k])[1] 
								for k in self.logs.keys()}
			time_delay = self.delay_func()
			#print('\t\t no activity for {} timesteps'.format(time_delay))
			for t in range(time_delay):
				self.timer += 1
				self.update_logs(old, self.timer)	
			self.demand_step(steps['d_step'])
			self.timer += 1/3
			self.supply_step(steps['s_step'])
			self.timer += 1/3
			old = {k : get_largest_key_val(self.logs[k])[1] 
								for k in self.logs.keys()}
			time_delay = self.delay_func()
			#print('\t\t no activity for {} timesteps'.format(time_delay))
			for t in range(time_delay):
				self.timer += 1
				self.update_logs(old, self.timer)	
			self.production_step(steps['p_step'])
			self.timer += 1/3
		#print('\tsimulation complete')

	def reset_simulation(self):
		self.logs =	{	'inventory' 	: {-0.5 : self.I0},
						'demand'		: {},
						'shortage'		: {},
						'supply'		: {},
						'production'	: {},
						'backlog'		: {},
						'reward_'	: { -0.5 : 0},
						'extra_inventory'	: { -0.5 : 0},
						'missed_reward'	: { -0.5 : 0}
					}
		self.demandQ = q.PriorityQueue()
		self.productionQ = q.PriorityQueue()
		self.supplyQ = q.PriorityQueue()
		self.timer = 0

	########################## steps ##########################

	def constant_demand_step(self):
		d = self.I0
		return {'demand' : d}

	def simple_demand_step(self):
		d = self.d_gen()
		return {'demand' : d}

	def all_or_nothing_supply_step(self):
		old = {k : get_largest_key_val(self.logs[k])[1] 
								for k in self.logs.keys()}
		if old['inventory'] >= old['demand']:
			supply = old['demand']
			new_inv = old['inventory'] - old['demand']
			shortage = 0
			excess = old['inventory'] - old['demand']
		else:
			supply = 0
			new_inv = old['inventory']
			shortage = old['demand'] - old['inventory']
			excess = old['inventory']
		new_csupply = old['reward_'] + supply
		new_cshortage = old['missed_reward'] + shortage
		new_cexcess = old['extra_inventory'] + excess
		return {'supply' : supply, 'inventory' : new_inv, 'shortage' : shortage,
				'reward_' : new_csupply, 'missed_reward' : new_cshortage, 
				'extra_inventory' : new_cexcess}
	
	def partial_order_supply_step(self):
		old = {k : get_largest_key_val(self.logs[k])[1] 
								for k in self.logs.keys()}
		if old['inventory'] >= old['demand']:
			supply = old['demand']
			new_inv = old['inventory'] - old['demand']
			shortage = 0
			excess = old['inventory'] - old['demand']
		else:
			supply = old['inventory']
			new_inv = 0
			shortage = old['demand'] - old['inventory']
			excess = 0
		new_csupply = old['reward_'] + supply
		new_cshortage = old['missed_reward'] + shortage
		new_cexcess = old['extra_inventory'] + excess
		return {'supply' : supply, 'inventory' : new_inv, 'shortage' : shortage,
				'reward_' : new_csupply, 'missed_reward' : new_cshortage, 
				'extra_inventory' : new_cexcess}

	def constant_inv_production_step(self):
		old = {k : get_largest_key_val(self.logs[k])[1] 
								for k in self.logs.keys()}
		current_inv = old['inventory']
		if current_inv >= self.I0:
			new_inv, production = current_inv, 0
		else:
			new_inv, production = self.I0, self.I0 - current_inv
		return {'inventory' : new_inv, 'production' : production}

	def bulk_production_step(self, delay):
		def func():
			old = {k : get_largest_key_val(self.logs[k])[1] 
									for k in self.logs.keys()}
			current_inv = old['inventory']
			if int(self.timer)%delay==0:
				if current_inv >= self.I0:
					new_inv, production = current_inv, 0
				else:
					new_inv, production = self.I0, self.I0 - current_inv
				return {'inventory' : new_inv, 'production' : production}
			else:
				return {'inventory' : current_inv, 'production' : 0}
		return func

	def ROP_production_step(self, ROP, ROO, delay=1):
		def func():
			old = {k : get_largest_key_val(self.logs[k])[1] 
									for k in self.logs.keys()}
			current_inv = old['inventory']
			if int(self.timer)%delay==0:
				if current_inv < ROP:
					new_inv, production = current_inv + ROO, ROO
				else:
					new_inv, production = current_inv, 0
				return {'inventory' : new_inv, 'production' : production}
			else:
				return {'inventory' : current_inv, 'production' : 0}
		return func


	####################### helper functions #######################

	def update_logs(self, results, time):
		for k, v in results.items():
			if k in self.logs.keys():
				if is_number(v):
					self.logs[k][time] = v
				else:
					print('cannot add:',v,'to',k,'logs')
			else:
				print('invalid log key:',k)

	def maxtime(self):
		mt = 0
		for k, v in self.logs.items():
			times = list(v.keys())
			times = times if len(times) != 0 else [0]
			mt = mt if mt > max(times) else max(times)
		return int(np.ceil(mt))

	def parse_steps(self, steps, kwargs):
		default_steps = {	'd_step' : self.simple_demand_step,
							's_step' : self.partial_order_supply_step,
							'p_step' : self.constant_inv_production_step,
							'strategy' : 'simple strategy'}
		if isinstance(steps, dict) and set(steps.keys()) == set(default_steps.keys()):
			return steps
		elif steps is None:
			strategy = kwargs.pop('strategy', None)
			if strategy is None:
				raise ValueError("exiting parse_steps() : no strategy or steps given")
			elif isinstance(strategy, str):
				strategy = ' '.join(strategy.split('_'))
				if strategy == 'simple strategy':
					self.I0 = kwargs.pop('I0', self.I0)
					steps = default_steps
				if strategy == 'bulk order strategy':
					steps = default_steps
					steps['strategy'] = strategy
					delay = kwargs.pop('delay', 5)
					steps['p_step'] = self.bulk_production_step(delay)
				if strategy == 'ROP strategy':
					steps = default_steps
					ROP = kwargs.pop('ROP', int(self.I0))
					ROO = kwargs.pop('ROO', 5*self.I0)
					self.ROP, self.ROO = ROP, ROO
					steps['strategy'] = strategy
					delay = kwargs.pop('delay', 1)
					steps['p_step'] = self.ROP_production_step(ROP, ROO, delay)
			return steps
		else:
			raise ValueError("exiting parse_steps() : invalid argument steps")

	def get_strategy_desc(self):
		try:
			strat = self.strategy
			self.strategy_desc = ' '.join(strat.split('_')) 
			if 'ROP' in self.strategy: self.strategy_desc = r'ROP$(R_0={},R={})$'.format(self.ROP, self.ROO)#+ r', $I_0={}$'.format(self.I0)
			self.strategy_foldername = '_'.join(strat.split(' '))
		except:
			raise Error('No strategy defined')

	###################### time delay ##############################

	def delay(self, delay_type=None, kwargs={}):
		txt = 'setting delay type as: '
		if delay_type == None:
			#print(txt+'0')
			return lambda : 0
		elif isinstance(delay_type, str):
			delay_type = delay_type.replace(' ', '_') 
		if delay_type == 'log_normal':
			mu = kwargs.pop('mu', np.log10(500))
			sigma = kwargs.pop('sigma', 5)	
			def func2():
				t = np.random.lognormal(mu, sigma)
				y2 = 60
				t = t if t < y2 else y2 + np.random.randint(-10, 20)
				return int(t)
				#print(txt+'log normal, mu={}, sigma={}'.format(mu, sigma))
			return func2

	###################### plotting functions ######################

	def plot_timeseries(self, showplot = True, savefig = False, **kwargs):
		results_dict = {k : v for k, v in self.logs.items() if v != {}}
		# get kwargs
		maxtime = kwargs.pop('maxtime', None) or max(list(self.logs['demand'].keys()))
		cm = kwargs.pop('cm', plt.cm.gnuplot)
		figsize = kwargs.pop('figsize', (14, 12))
		scatterkeys = kwargs.pop('scatterkeys', list(results_dict.keys()))
		y_lims = kwargs.pop('y_lims', {}).copy()
		plotorder = kwargs.pop('plotkeys', [['demand'], ['inventory'], 
											['supply'], ['production'],
											['shortage'], ['reward_', 'missed_reward', 'extra_inventory']])
		# get cmap
		col = iter(cm(i/len(results_dict)) for i in range(len(results_dict)))
		# setup plots
		plt.close('all')
		fig, axs = plt.subplots(int(np.ceil(len(plotorder)/2)), 2, sharex='col', figsize = figsize)
		axs = iter(j for i in axs for j in i)
		allmax = {ind : max(vals.values()) for ind, vals in results_dict.items()}
		check_lims = {k : False for k in results_dict}
		for plotlist in plotorder:
			ax = next(axs)
			for k in plotlist:
				v = results_dict[k]
				to_scatter, c = k in scatterkeys, next(col)
				single_timeseries(ax, v, k, to_scatter, col=c, maxtime=maxtime)
				if k == 'inventory' and 'ROP' in self.strategy:
					ax.axhline(self.ROP, c='grey', alpha=0.7, label=r'$R_0$')
					ax.legend(loc='upper left', fontsize=21)
				if y_lims and isinstance(y_lims, dict):
					check_lims[k] = y_lims.pop(k, False)
					if check_lims[k]:
						ax.set_ylim(check_lims[k])
			#ax.set_xlabel('time')
			ax.set_ylim(ymin=0)
			ax.set_xlim((0, maxtime))
			"""
			if not any([check_lims[a] for a in plotlist]):
				if any([a in ['supply', 'production'] for a in plotlist]):
					ax.set_ylim((0, 1.1*roundup(max([allmax[k] for k in ['supply', 'production']])) ) )
				elif any([a in ['demand', 'inventory'] for a in plotlist]):
					ax.set_ylim((0, 1.1*(max([allmax[k] for k in ['demand', 'inventory']]))))
				elif not any(['_' in a for a in plotlist]):
					ax.set_ylim((0, 1.05*max(allmax.values())))
			"""
			# legend or title
			if len(plotlist) > 1:
				lgnd = ax.legend(fontsize=22, loc=2)
				for i in range(len(lgnd.legendHandles)):
					lgnd.legendHandles[i]._sizes = [45]
			else:
				lab = k
				if lab == 'demand':
					lab = 'orders, '+self.d_gen.desc
				if lab == 'inventory':
					lab += r', $I_0={}$'.format(self.I0)
				if k == 'supply':
					lab = 'shipped, '+self.strategy_desc
				if k == 'production':
					lab = 'restocked'
				ax.set_title(lab)
		# title stuff
		titletext = ''
		# titletext += 'timeseries for {} eventsteps\n'
		#titletext += 'demand: '.format(maxtime)+str(self.d_gen.desc)
		#titletext += '\nstrategy: {}'.format(self.strategy)+', {}'.format(self.strategy_desc)
		#plt.suptitle(titletext, y = 0.98, weight = 'bold')
		# layout stuff
		plt.tight_layout(rect=[0, 0.03, 1, 0.98])
		# show or save
		if showplot:
			plt.show()
		else:
			savechoice = savefig or input('save timeseries?\t y/n\t') == 'y'
			if savechoice:
				foldername = join(self.strategy_foldername, self.d_gen.foldername, 'timeseries')
				os.makedirs(join(self.cwd, foldername), exist_ok=True)
				savepath = join(self.cwd, foldername, 'I0={},time={}.png'.format(self.I0, maxtime))
				fig.savefig(savepath)
				print('\t\tsaved', savepath)

	def plot_histograms(self, logscale = False, showplot = True, to_remove = {}, savefig = False, **kwargs):
		results_dict = {k : v for k, v in self.logs.items() if v != {}}
		# get kwargs
		bins = kwargs.pop('bins', {k:50 for k in results_dict.keys()})
		annotate = kwargs.pop('annotate', {k:False for k in results_dict.keys()})
		if isinstance(annotate, bool):
			annotate = {k:annotate for k in results_dict.keys()}
		x_lims = kwargs.pop('x_lims', {}).copy()
		check_lims = {k : False for k in results_dict}
		cm = kwargs.pop('cm', plt.cm.gnuplot)
		figsize = kwargs.pop('figsize', (14, 12))
		scatterkeys = kwargs.pop('scatterkeys', list(results_dict.keys()))
		plotorder = kwargs.pop('plotkeys', [['demand'], ['inventory'], 
											['supply'], ['production'],
											['shortage'], ['missed_reward', 'extra_inventory', 'reward_']])
		# setup figure
		plt.close('all')
		fig, axs = plt.subplots(int(np.ceil(len(plotorder)/2)), 2, figsize = figsize)
		axs = iter(j for i in axs for j in i)
		# setup colors
		col = iter(cm(i/len(results_dict)) for i in range(len(results_dict)))
		## plot!
		for plotlist in plotorder:
			ax = next(axs)
			if any(['_' in k for k in plotlist]):
				ax.axis('off')
				continue
			for k in plotlist:
				if x_lims and isinstance(x_lims, dict):
					check_lims[k] = x_lims.pop(k, False)
					if check_lims[k]:
						ax.set_xlim(check_lims[k])
				v = results_dict[k]
				c = next(col)
				single_histogram(ax, v, k, to_remove, col=c, bins=bins[k], annotate=annotate[k])
				ax.set_ylim(0, 1)
			#
			lab = ', '.join([k.replace('_', ' ') for k in plotlist]) #+ ' (sample size={})'.format(numvals)
			if lab == 'demand':
				lab = 'orders, '+self.d_gen.desc
			if lab == 'inventory':
				lab += r', $I_0={}$'.format(self.I0)
			if k == 'supply':
				lab = 'shipped, '+self.strategy_desc
			if k == 'production':
				lab = 'restocked'
			ax.set_title(lab)
			ax.set_xlim(0)
			#ax.set_xlabel('values of '+lab.split('(')[0])
			if not any([l in ['inventory', 'production'] for l in plotlist]):
				ax.set_ylabel(r'$P$')
			#lgnd = plt.legend()
			#for i in range(len(lgnd.legendHandles)):
			#		lgnd.legendHandles[i]._sizes = [45]
			#plt.gca().set_major_locator(ticker.MaxNLocator(integer=True))
		#titletext = ''
		# titletext += 'distribution for {} eventsteps\n'
		#titletext += 'demand: '.format(self.maxtime())+str(self.d_gen.desc)
		#titletext += '\nstrategy: {}'.format(self.strategy)+', {}'.format(self.strategy_desc)
		#plt.suptitle(titletext, y = 0.98, weight = 'bold')
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		# show or save
		if showplot:
			plt.show()
		else:
			savechoice = savefig or input('save timeseries?\t y/n\t') == 'y'
			if savechoice:
				foldername = join(self.strategy_foldername, self.d_gen.foldername, 'distribution')
				os.makedirs(join(self.cwd, foldername), exist_ok=True)
				savepath = join(self.cwd, foldername, 'I0={},time={}.png'.format(self.I0, self.simtime))
				fig.savefig(savepath)
				print('\t\tsaved', savepath)

#################### plot 1 thing on an axis #####################################

def single_histogram(ax, data_dict, label, to_remove = {}, logscale = False, **kwargs):
	col = kwargs.pop('col', 'k')
	bins = kwargs.pop('bins', 50)
	figsize = kwargs.pop('figsize', (10, 5))
	annotate = kwargs.pop('annotate', False)
	if ax is None:
		plt.figure(figsize = figsize)
		ax = plt.gca()
	#
	removals = {}
	if isinstance(data_dict, dict):
		data = list(data_dict.values())
	else:
		data = list(data_dict)
	#
	numvals = len(data)
	data2 = data.copy()
	if label in to_remove.keys():
		removals = {(label, t) : data.count(t) for t in to_remove[label]}
		data2 = np.array([i for i in data if i not in to_remove[label]])
	if logscale:
		data2 = np.log(data2)
	# begin histogram calculations
	data2 = [val for val in data2 if ~np.isnan(val)]
	counts, edges = np.histogram(data, bins)
	vals = [np.mean(edges[i:i+2]) for i in range(len(edges) - 1)]
	#print(label, len(vals), len(counts))
	s1 = {key : val for key, val in zip(vals, counts) if val != 0}
	if len(s1) != 0:
		vals, counts = zip(*s1.items())
		counts = counts/ np.sum(counts)
		#print(label, vals, counts)
		ax.stem(vals, counts, color = col, label = label.replace('_', ' ')+' (sample size={})'.format(numvals), s=7)
	#ymax = min([1, hist_roundup(max(counts))])
	ax.set_ylim(0, 0.4)
	#print('\t\t\tsetting ylim 0.4')
	# tick formatting
	ax.set_yticks(list(np.arange(0, 1.1, 0.2)))
	ax.set_yticklabels(list(np.arange(0, 1.1, 0.2)))
	#ax.ticklabel_format(style='sci', axis='y', scilimits=(2,0))
	#ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
	# adding annotation
	notes = '\n'.join([r'$P({})$ = {}'.format(int(k1[1]), format(v1/numvals, '.1f')) 
						for k1, v1 in removals.items()])
	if annotate:
		ax.text(0.5, 0.97, notes, ha = 'center', va = 'top', transform=ax.transAxes, fontsize = 14)
	#if label == 'demand':
	#	ax.axvline(np.mean(data), c = 'k', alpha = 0.3)

def single_timeseries(ax, data_dict, label, to_scatter = False, **kwargs):
	col = kwargs.pop('col', 'k')
	s, alpha = kwargs.pop('s', 10), kwargs.pop('alpha', 1)
	maxtime = kwargs.pop('maxtime', np.inf)
	y_lims = kwargs.pop('y_lims', None)
	#
	data_dict = {time : val for time, val in data_dict.items() if time <= maxtime}
	if y_lims:
			ax.set_ylim(y_lims)
	else:
		ax.set_ylim(ymin=0)
	# special code for inventory plots
	"""	
	if label == 'inventory':
		remove = {k : v for k, v in data_dict.items() if check_time(k, 1/3)}
		refill = {k : v for k, v in data_dict.items() if check_time(k, 2/3)}
		ax.step(remove.keys(), remove.values(), label = 'depleted', color = 'red', alpha = alpha, lw=1.5)
		ax.step(refill.keys(), refill.values(), label = 'replenished', color = plt.cm.Blues(0.8), alpha = alpha, lw=2.5)
		#ax.scatter(remove.keys(), remove.values(), label = 'inv after supply', color = 'red', s = s, alpha = alpha, lw = 0.2, edgecolors = 'k')
		#ax.scatter(refill.keys(), refill.values(), label = 'inv after production', color = plt.cm.Blues(0.8), s = s, alpha = alpha, lw = 0.2, edgecolors = 'k')
		if max([max(remove.values()), max(refill.values())]) > 10000:
			ax.set_yticks([10000,20000, 30000, 40000, 50000, 60000, 70000])
			ax.set_yticklabels(['10K','20K', '30K', '40K', '50K', '60K', '70K'])
		lgnd = ax.legend(fontsize='xx-large', loc='upper left')
		for i in range(len(lgnd.legendHandles)):
			lgnd.legendHandles[i]._sizes = [45]
		return 0
	"""
	if isinstance(data_dict, dict):
		times = np.array(list(data_dict.keys()))
		vals = np.array(list(data_dict.values()))
	if times.size == 0:
		print('not plotting key:', label.replace('_', ' '))
		return 0
	#print(len(times), len(vals), label, times[0], vals[0])
	if to_scatter:
		ax.scatter(times, vals, label = label.replace('_', ' '), color = col, s = s, alpha = alpha, lw = 0.2, edgecolors = 'k')	
	else:
		ax.step(times, vals, label = label.replace('_', ' '), color = col, alpha = alpha, lw=1.8)
		#ax.plot(times, vals, label = label.replace('_', ' '), color = col, alpha = alpha, lw=3)
		#ax.scatter(times, vals, color = 'k', s = 5, alpha = 0.7)
	#if label == 'demand':
	#	ax.axhline(np.mean(vals), c = 'k', alpha = 0.3)

#################### helper functions ##########################

def get_largest_key_val(data_dict):
	k = max(data_dict.keys())
	return k, data_dict[k]


def is_number(s):
	try:
		float(s)
		return True
	except:
		return False

#####################################################################

class DemandGenerator:
	def __init__(self, d, desc, minmax=(-np.inf, np.inf)):
		assert callable(d), 'provide valid demand function'
		self.d_gen = d
		self.desc = desc
		self.minmax = minmax
		self.foldername = '_'.join(desc.replace('$', "").replace("_", "").replace('\\', "").split(' '))
		self.demand_type = desc.split()[0] if desc.split()[0] in ['normal', 'powerlaw', 'uniform'] else desc
	#
	def __call__(self):
		d = self.d_gen()
		m1, m2 = self.minmax
		i = 0
		while d < m1 or d > m2:
			#print(d)
			d = self.d_gen()
			i += 1
			if i % 5000 == 0:
				print('trying to simulate demand within range - try #', str(i))
		return d

def get_normal_DemandGenerator(mu, sigma, minmax=(0, np.inf)):
	"""	returns DemandGenerator with N(mu, sigma)
	"""
	assert all([arg is not None for arg in (mu, sigma)])
	d = lambda : mu + sigma * np.random.randn()
	d_gen = DemandGenerator(d, r'$N(\mu={},\sigma={})$'.format(mu, sigma), minmax)
	d_gen.mu, d_gen.sigma = mu, sigma
	return d_gen

def get_powerlaw_DemandGenerator(alpha, mu, minmax=(-np.inf, np.inf)):
	"""	returns DemandGenerator as (1-k)^(1/(1-alpha))
	"""
	assert all([arg is not None for arg in (alpha, mu)])
	from scipy.stats import beta
	b = alpha * (mu - 1)
	d = lambda : np.power((beta.rvs(alpha, b)), -1) 
	d_gen = DemandGenerator(d, r'powerlaw ($\alpha={},\mu={})$'.format(format(alpha, '.2f'), int(mu)), minmax)
	d_gen.alpha = alpha
	return d_gen

def get_uniform_DemandGenerator(a, b):
	"""	returns DemandGenerator as a + k*(b - a)
	"""
	assert all([arg is not None for arg in (a, b)])
	d = lambda : a + np.random.random() * (b - a)
	d_gen = DemandGenerator(d, r'uniform in $[{},{})$'.format(a, b), minmax=(a, b))
	d_gen.a, d_gen.b = a, b
	return d_gen

##################### useful

def check_time(t, r):
	t = float("{0:.2f}".format(t))
	r  = float("{0:.2f}".format(r))
	res = float("{0:.10f}".format((t%1)-r))
	return res == 0.0

def roundup(x, unit = 100):
	return int(np.ceil(x / unit)) * int(unit)

def hist_roundup(x, multiple = 0.2):
	newx = np.ceil(np.round(x*20, decimals = 2))/10
	return newx


###################################################



"""
i = Inventory_MGMT(I0 = 100, d_gen = get_powerlaw_DemandGenerator(1.05, 500))

i.run_simulation(eventsteps = t)
i.plot_timeseries(showplot = False, savefig = sf)
i.plot_histograms(showplot = False, savefig = sf)

i = Inventory_MGMT(I0 = 100, d_gen = get_powerlaw_DemandGenerator(1.5, 150))

i.run_simulation(eventsteps = t)
i.plot_timeseries(showplot = False, savefig = sf)
i.plot_histograms(showplot = False, savefig = sf)

i = Inventory_MGMT(I0 = 10, d_gen = get_normal_DemandGenerator(mu=100, sigma=50))

i.run_simulation(eventsteps = 1000)
i.plot_timeseries(showplot = True, savefig = sf)
i.plot_histograms(showplot = True, savefig = sf)
"""

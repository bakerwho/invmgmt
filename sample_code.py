import invmgmt

t = 5000

sim = invmgmt.Inventory_MGMT(demand_type='powerlaw', mu=100, alpha=2.6)
sim.run_simulation(timesteps = t, strategy='ROP_strategy', ROP=200, ROO=1500)

times, vals = sim.logs['inventory'].keys(), sim.logs['inventory'].values()
times, vals = list(times), list(vals)


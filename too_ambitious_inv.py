
class Order:
	def __init__(self, material, qty, unitid = 1):
		assert qty > 0
		self.material = material
		self.quantity = qty
		self.unitid = unitid

class Production:
	def __init__(self, material, qty, unitid = 1):
		assert qty > 0
		self.material = material
		self.quantity = qty
		self.unitid = unitid

class Warehouse:
	def __init__(self, WHname, WHcoords = None):
		self.name = WHname
		self.WHcoords = WHcoords
		self.stock = {}

	def add_material(self, Production):
		if Production.unitid == 1:
			if Production.material in self.stock.keys():
				self.stock[Production.material] += Production.quantity
			else:
				self.stock[Production.material] = Production.quantity
		else:
			raise ValueError('Invalid unitid')

	def remove_material(self, Order):
		mtl, qty = Order.material, Order.quantity
		if Order.unitid == 1:
			if mtl in self.stock.keys() and self.stock[mtl] >= qty:
				self.stock[mtl] -= qty
		else:
			raise ValueError('Invalid unitid')


def Company:
	def __init__(self, WHdict = {}):
		assert isinstance(WHdict, dict)
		self.WHdict = WHdict
		self.WHcount = len(WHdict)
		self.Orders = {}
		self.time = 0
		self.history = WHdict

	def add_WH(self, WH):
		assert isinstance(WH, Warehouse)
		self.WHdict[WH.name] = WH
		self.WHcount += 1

	def get_WH_stock(self):
		self.global_stock = { WH.name : WH.stock for WH in WHdict.items()}
		return self.global_stock

	def get_material_stock(self):
		matl_stock = {}
		for WH in WHdict.items():
			for mat, qty in WH.stock:
				if mat in matl_stock.keys():
					matl_stock[mat] += qty
				else:
					matl_stock[mat] = qty
		self.material_stock = matl_stock
		return matl_stock

	def receive_Order(self, material, qty, unitid = 1):


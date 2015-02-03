bl_info = {
	"name": "Tree Generator",
	"author": "Nellie Robinson",
	"version": (0,1),
	"blender": (2, 67, 0),
	"location": "View3D > Add > Mesh",
	"description": "Adds Tree",
	"category": "Object"}

import bpy
import bmesh
from bpy.props import IntProperty, FloatProperty, BoolProperty, FloatVectorProperty, CollectionProperty
from random import uniform, random, getrandbits
from mathutils import Vector, Matrix   
from math import pi, cos, sin, acos, asin, sqrt, fabs, degrees, radians
from copy import copy

# creates spline (branch segment)
def makeSpline(cu, typ, points, weights):
	spline = cu.splines.new(typ)
	npoints = len(points)
	if typ == 'BEZIER' or typ == 'BSPLINE':
		spline.bezier_points.add(npoints-1)
		for (n,pt) in enumerate(points):
			bez = spline.bezier_points[n]
			(bez.co, bez.handle1, bez.handle1_type, bez.handle2, bez.handle2_type) = pt
	else:
		spline.points.add(npoints-1)    # One point already exists?
		for (n,pt) in enumerate(points):
			spline.points[n].co = pt
			spline.points[n].radius = weights[n]/ 20
	spline.use_endpoint_u = True
	return

# calculates inner product of vector
def inner_product(v):
	return sqrt(sum([x**2 for x in v]))

# calculates norm of vector
def norm(v):
	return v / inner_product(v)

# returns random vector
def rand_vec(minimum, maximum, length):
	return Vector([uniform(minimum, maximum) for l in range(length)])

# returns matrix of rotation
def rotation_mat(axis, angle):
	c = cos(angle)
	s = sin(angle)
	t = 1-c
	x = axis.x
	y = axis.y
	z = axis.z
	return Matrix([[t*(x**2)+c, t*x*y - s*z, t*x*z + s*y],
				[t*x*y + s*z, t*(y**2)+c, t*y*z - s*x],
				[t*x*z - s*y, t*y*z + s*x, t*(z**2) + c]])

# multiplies vector by scalar
def vector_mult(a, b):
	return Vector([b * x for x in a])

# multiplies two vectors
def vec_vec_mult(a, b):
	return Vector([x * y for (x, y) in zip(a, b)])


class Branch(object):
	
	def __init__(self, co = Vector([0, 0, 0]), a = Vector([0,0,1]), l = 10, w = 20):
		self.co = co
		self.a = norm(a)
		self.l = l
		self.w = w
	
	# testing purposes
	def __str__(self):
		return "{}, {}, {}, {}".format([x for x in self.co],self.w,self.l,[degrees(x) for x in self.a])


class TreeGenerator(bpy.types.Operator):
	bl_idname = "primitive_mesh.tree_add"
	bl_label = "Generate Tree"
	bl_options = {'REGISTER', 'UNDO'}


	l0 = FloatProperty(
		name = "length",
		description = "length of initial branch",
		default = 10,
		min = 1,
		max = 50,
	)
	
	w0 = FloatProperty(
		name = "weight",
		description = "weight of initial branch",
		default = 20,
		min = 1,
		max = 100,
	)

	lr = FloatProperty(
		name = "shortening rate",
		description = "rate at which length decreases",
		default = 1.109,
		min = 0.5,
		max = 1.5,
	)

	vr = FloatProperty(
		name = "weight decrease rate",
		description = "rate at which weight decreases",
		default = 1.732,
		min = 1.2,
		max = 2,
	)

	nr = IntProperty(
		name = "segments per branch",
		description = "used for curving branches",
		default = 3,
		min = 2,
		max = 5,
	)

	ba1 = FloatProperty(
		name = "max",
		description = "max branching angle",
		default = pi/3,
		min = pi/4,
		max = pi/2,
	)

	ba2 = FloatProperty(
		name = "min",
		description = "min branching angle",
		default = pi/12,
		min = pi/16,
		max = pi/4,
	)

	prune = BoolProperty(
		name = "prune",
		description = "cuts off branches beyond certain radius",
		default = False,
	)

	prune1 = FloatVectorProperty(
		name = "pruning point 1",
		default = (0, 0, 10, 20),
		size = 4,
	)
	
	prune2 = FloatVectorProperty(
		name = "pruning point 2",
		default = (0, 0, 10, 20),
		size = 4,
	)

	prune3 = FloatVectorProperty(
		name = "pruning point 3",
		default = (0, 0, 10, 20),
		size = 4,
	)

	wr = FloatProperty(
		name = "lumpiness",
		description = "random variations in branch weight",
		default = 0,
		min = 0,
		max = 1,
	)

	cr = FloatProperty(
		name = "curvature",
		description = "random variations in vertices of branch segments",
		default = 0,
		min = 0,
		max = 1,
	)


	twig_threshold = FloatProperty(
		name = "twig threshold",
		description = "required weight of branch to stop recursion",
		default = 0.6,
		min = 0.2,
		max = 3.0,
	)

	gradual_angle = FloatVectorProperty(
		name = "gradual angle",
		description = "angles branches towards certain angle",
		size = 3,
		default = (1, 1, 1),
		min = 0.5,
		max = 1.5,
	)
		
	gradual_scale = FloatVectorProperty(
		name = "gradual scale",
		description = "scales branches in certain dimensions",
		size = 3,
		default = (1, 1, 1),
		subtype = 'XYZ',
		min = 0.5,
		max = 1.5,
	)

	straight_branches = BoolProperty(
		name = "continuous branches",
		description = "whether the main branch in intersection curves",
		default = False,
	)

	randomize = BoolProperty(
		name = "randomize",
		description = "randomizes parameters (may produce slow and/or unexpected results",
		default = False,
	)


	# randomizes parameters to tree generator
	def randomize_all(self):
		self.l0 = uniform(5, 50)
		self.w0 = uniform(5, 80)
		self.lr = uniform(.6, 1.4)
		self.vr = uniform(1.3, 2)
		self.nr = uniform(2, 5)
		self.ba1 = uniform(pi/4, pi/2)
		self.ba2 = uniform(pi/16, pi/4)
		self.wr = uniform(0, 1)
		self.cr = uniform(0, 1)
		self.twig_threshold = uniform(0.5, 3.0)
		self.gradual_scale = rand_vec(.6, 1.4, 3)
		self.gradual_angle = rand_vec(.6, 1.4, 3)

	# returns points and weights of a curved branch
	def curve_branch(self, co1, co2, w1, w2, n):
		diff = co2 - co1
		diff_w = w2 - w1
		return ([Vector([co1.x + (i/(n - 1)) * diff.x * uniform(self.cr, self.cr2),
						 co1.y + (i/(n - 1)) * diff.y * uniform(self.cr, self.cr2),
						 co1.z + (i/(n - 1)) * diff.z * uniform(self.cr, self.cr2), 
						 1])
				 for i in range(n)],
				[(w1 + (i/(n - 1)) * diff_w) * (uniform(self.wr, self.wr2) if i != 0 and i != n-1 else 1)
				 for i in range(n)])

	# draws spline and returns its endpoint
	def forward(self, br):
		target = copy(br)
		target.co = br.co + vec_vec_mult(vector_mult(br.a, br.l), self.gradual_scale)
		target.w = br.w / self.vr 
		(coords, weights) = self.curve_branch(br.co, target.co, br.w, target.w, self.nr)
		makeSpline(cu, "NURBS", coords, weights)
		target.co = Vector([coords[self.nr-1].x, coords[self.nr-1].y, coords[self.nr-1].z])
		return target

	# returns next branch 
	def calcBranch(self, branch, angle=0):
		rotation = uniform(0, pi * 2)
		br = copy(branch) 
		a = br.a
		o = Vector([uniform(-1,1),uniform(-1,1),uniform(-1,1)]) 
		if a.x != 0:
			o.x = -(a.y * o.y + a.z * o.z) / a.x
		elif a.y != 0:
			o.y = -(a.x * o.x + a.z * o.z) / a.y
		elif a.z != 0:
			o.z = -(a.x * o.x + a.y * o.y) / a.z
		else:
			raise Exception("Invalid input: zero vector")
		o = norm(o)
		assert(inner_product(o) > .9999 or inner_product(o) < 1.0001)
		assert(o * a < 0.0001)
		br.a = rotation_mat(branch.a, rotation) * (rotation_mat(o, angle) * br.a)
		br.a = norm(vec_vec_mult(br.a, self.gradual_angle))
		return br

	# whether this branch should be pruned
	def to_prune(self, coord):
		for (c, n) in pruning_pts:
			if (coord - c).length < n:
				return False
		return True

	# returns tuple consisting of distance from farthest threshold and radius of said threshold
	def min_prune(self, coord):
		return max([(n - (coord - c).length, n) for (c, n) in pruning_pts if (coord - c).length < n])
			
	# main function; draws branch and keeps recurring
	def branch(self, br):
		if (br.w > self.twig_threshold and (not self.prune or not self.to_prune(br.co))):
			if self.prune:
				temp = self.min_prune(br.co)
				temp1 = (temp[0]) / temp[1] +0.5
				temp1 = temp1 if (temp1 <= 0.9) else 0.9
				br.l *= temp1
			else:
				br.l /= (self.lr * uniform(0.9, 1.2))
			target = self.forward(Branch(br.co, br.a, br.l, br.w))
			choice = random()
			if (choice > 0.05):
				self.branch(self.calcBranch(branch=target, angle = uniform(self.ba1, self.ba2)))
			if (choice > 0.2):   
				self.branch(self.calcBranch(branch=target, angle = uniform(self.ba1, self.ba2)))
			if (self.straight_branches):
				self.branch(self.calcBranch(branch=target, angle=0))
			else:
				self.branch(self.calcBranch(branch=target, angle=uniform(0, self.ba1/2)))

	# menu layout
	def draw(self, context):
		layout = self.layout
		# initial characteristics
		box3 = layout.box()
		box3.label("Intial")
		col = box3.column()
		col.prop(self, 'l0')
		col.prop(self, 'w0')
		# recursive characteristics
		box4 = layout.box()
		box4.label("Recursive")
		box4.prop(self, 'lr')
		box4.prop(self, 'vr')
		box4.prop(self, 'nr')
		box4.prop(self, 'twig_threshold')
		row1 = box4.row(align = True)
		row1.label("angle")
		row1.prop(self, 'ba1')
		row1.prop(self, 'ba2')
		box4.prop(self, 'straight_branches')
		# pruning
		box = layout.box()
		box.label("Pruning")
		box.prop(self, 'prune')
		box.prop(self, 'prune1')
		box.prop(self, 'prune2')
		box.prop(self, 'prune3')
		# variation
		box1 = layout.box()
		box1.label("Variation")
		box1.prop(self, 'wr')
		box1.prop(self, 'cr')
		# asymptotic
		box2 = layout.box()
		box2.label("Skew")
		box2.prop(self, 'gradual_angle')
		box2.prop(self, 'gradual_scale')
		layout.prop(self, 'randomize')

	# when the operator is applied
	def execute(self, context):
		global cu
		global ob
		global scn
		cu = bpy.data.curves.new("MyCurve", "CURVE")
		ob = bpy.data.objects.new("MyCurveObject", cu)
		scn = bpy.context.scene
		init_loc = bpy.context.scene.cursor_location
		scn.objects.link(ob)
		scn.objects.active = ob
		cu.dimensions = "3D"
		cu.fill_mode = 'FULL'
		cu.bevel_depth = 1.0
		cu.bevel_resolution = 3

		if (self.randomize):
			self.randomize_all()
			self.randomize = False

		self.cr += 1
		self.wr += 1
		self.cr2 = 1 - self.cr 
		self.wr2 = 1 - self.wr 

		if self.prune:
			global pruning_pts
			pruning_pts = [(init_loc + Vector([a, b, c]), d) 
					for (a, b, c, d) in {self.prune1, self.prune2, self.prune3}]
		starter = Branch(co = init_loc, l = self.l0, w = self.w0)
		self.branch(starter) 
		return {'FINISHED'}

def menu_func(self, context):
	self.layout.operator(TreeGenerator.bl_idname, text="Tree", icon='PLUGIN')

def register():
	bpy.utils.register_module(__name__)
	bpy.types.INFO_MT_mesh_add.append(menu_func)

def unregister():
	bpy.utils.unregister_module(__name__)
	bpy.types.INFO_MT_mesh_add.remove(menu_func)

if __name__ == "__main__":
	register()

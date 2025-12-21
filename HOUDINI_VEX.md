Houdini VEX doc

https://www.sidefx.com/docs/houdini/vex/lang.html
https://www.sidefx.com/docs/houdini/vex/snippets.html
https://www.sidefx.com/docs/houdini/vex/functions/index.html
_________________________________________



Instantiating a vector:
	
	vector up = { 0, 1, 0 };

Attribute syntax:

	@P  			// current point position
	@ptnum 			// index of the current point (when iterating over points)
	@primnum			// index of the current primitive (when iterating over primitives)


`geoself()`: handle to current geometry


Modifying geometry:
	
	int  addprim(int geohandle, string type) 					// returns primitive number
	int  addpoint(int geohandle, int point_number) 				// add a copy of point number
	int  addpoint(int geohandle, vector pos) 
	int  addvertex(int geohandle, int prim_num, int point_num)	// add point as vertex in current geometry
	int  setpointattrib(int geohandle, string name, int point_num, <type>value, string mode="set")
	int  setpointattrib(int geohandle, string name, int point_num, <type>value[], string mode="set")
	int  setprimgroup(int geohandle, string name, int prim_num, int value, string mode="set")


To set only a component of a point attrib:
	
	value = getpointattrib(geohandle, name, point_num);
	value.component = ...;
	setpointattrib(geohandle, name, point_num, value);

Foreach vertex point in primitive: 
	int[] vertices = primvertices();
	foreach (int vtx; vertices) {
	}


where `type_string` can be `"polyline"`, ... 

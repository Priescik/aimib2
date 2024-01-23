function addTheaterObject()
{
        var o = Obj3D.load("theater");
        o["swiatlol"].transparent=1;
        o["swiatlop"].transparent=1;
	var e = Obj3D.newEmpty();
	e.add(o).transform = Matrix.newScale(0.3);
	GLDisplay.world3d.add(e);
	return e;
}

function createGlassMaterial()
{
	var mat = Material.new();
	mat.lighting = 0;
	mat.color = 0x446666ff;
	mat.texture = "whiteworld_env_opt:m1"; //TEXTURE_MIN_FILTER=GL_LINEAR
	mat.shader = "glassreflection";
	return mat;
}

function addBottleObject()
{
	var o = Obj3D.newEmpty();
	o.add(Obj3D.load("demoworld")).transform = Matrix.newTranslation((0, 0, -0.4));
	o.add(Obj3D.load("coverlabel")).transform = Matrix.newTranslation((0, 0, -0.4));
	var cover;
	cover = Obj3D.load("coverhole");
	o.add(cover).transform = Matrix.newTranslation((0, 0, -0.4));
	cover.transparent = 1;
	cover.material = createGlassMaterial();

	GLDisplay.world3d.add(o);
	return o;
}

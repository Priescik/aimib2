function default_neuro_build()
{
  Element.useNeuro();
  if (n.class=="Light")
{
  GeomBuilder.addBranch(GeomBuilder.currentNode);
  var br=GeomBuilder.currentNode;
  Element.node=br;
  Loader.load("light-base.ac");
  if (Loader.loaded) GeomBuilder.add(Loader.loaded);
  GeomBuilder.addTransform(br);
  Loader.load("light-on.ac");
  if (Loader.loaded) GeomBuilder.add(Loader.loaded);
}
  else
{
  var filename;
  if (n.class == "|")
    filename = "neuro-bend.ac";
  else
    filename = "neuro-" + n.class + ".ac";
  Loader.load(filename);
  if (Loader.loaded)
    {
    GeomBuilder.addTransform(GeomBuilder.currentNode);
    GeomBuilder.matrixReset();
    if (n.liveNeuro)
      {
      if (n.class == "Gpart")
        {
        var part=n.liveNeuro.mechpart.part;
	if (part.sh!=0)
          GeomBuilder.matrixScale(part.sx*5,part.sy*5,part.sz*5);
        }
      GeomBuilder.matrixMove1(n.liveNeuro.relative_pos);
      GeomBuilder.matrixRotate1(n.liveNeuro.relative_orient);
      if (n.class == "Gpart")
        GeomBuilder.matrixMove(-0.5,0,0);
      GeomBuilder.setMatrix(GeomBuilder.currentNode);
      }
    GeomBuilder.add(Loader.loaded);
    }
 }
}

function default_neuro_update()
{
  Element.useNeuro();
  if (n.class=="Light")
	{
	var tr=GeomBuilder.getBranch(Element.node,1);
	if (tr)
		{
		GeomBuilder.matrixReset();
		var sc=n.state;
		if (sc<0.1) sc=0.1; else if (sc>1.0) sc=1.0;
		GeomBuilder.matrixScale(sc,sc,sc);
		GeomBuilder.setMatrix(tr);
		}
	}
}

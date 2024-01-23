global g_hide_screenshot;

function handleSharing(elem)
{
	if (elem.name == "share")
	{
		Sharing.select("Share the current view and/or your creature's genotype:", function sharing_function);
		return 1;
	}
	return 0;
}

function sharing_function()
{
	Simulator.export("share.gen", 2, 0, 0);
	Sharing.attach_file = "scripts_output" + File.pathseparator + "share.gen";

	var edgebuttons = ["top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"];
	for(var name in g_hide_screenshot) GUI[name].visible = 0;
	for(var name in edgebuttons) GUI.[name].visible = 0;
	var lab = GUI.newLabel("", "www.framsticks.com");
	lab.setAutolayout(0, 0, 0, 0, 0, 0, 1, 1);
	lab.align = 1;
	lab.valign = 1;
	lab.fgcolor = 0x66000000;
	Sharing.addScreenshot();
	for(var name in g_hide_screenshot) GUI[name].visible = 1;
	for(var name in edgebuttons) GUI.[name].visible = 1;
	GUI.removeElement(lab);

	Simulator.print("sharing:");
	Simulator.print(Sharing.service);
	Simulator.print(Sharing.message);
	Simulator.print(Sharing.url);
}

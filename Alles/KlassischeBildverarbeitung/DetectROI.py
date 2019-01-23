extra_pixels = 9;
initial_blur = 2;
median =  10;threshold = "Default";

// Segment image

roiManager("Reset");
run("Duplicate...", "title=detector");
run("Median...", "radius=2"); // remove line artifacts...
run("16-bit");
run("Gaussian Blur...", "sigma="+initial_blur);
run("Find Edges");
run("Median...", "radius="+median);
setAutoThreshold(threshold+" dark");


// Extract detected particles
run("Analyze Particles...", "add");



// Convert from shape to bounding box and enlarge
c = roiManager("count");

for(i=0; i<c;i++) {
	roiManager("Select", i);
	run("To Bounding Box");
	run("Enlarge...", "enlarge="+extra_pixels+" pixel");
	Roi.setName("Region #"+IJ.pad(i+1,2));
	roiManager("Add");
}

// delete original ROIs
roiManager("Select", Array.getSequence(c));
roiManager("Delete");
run("Select None");

close();
roiManager("Show All with labels");
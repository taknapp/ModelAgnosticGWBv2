Any script with "_analysis" will produce a diagnostic or output plot. 
Any script with "_fitter" will do the sampling for a given signal and noise. 
In each script, you can use keywords "classic", "Sachdev1e", "FOPT" to indicate the signal injected. These signals are defined in the helper script. 
You can also use "HL" or "CE" to choose either A+ or CE noise. These should be automatically implemented based on the variables defined at the beginning of the script. 
You can also define the observing times you want to run for. These are in an array at the top of the script. I believe they're in seconds but can easily be converted and I try to label with units when I go between seconds and years. 

These aren't super pretty or clean yet but they work pretty smoothly. 
The output from each fitter or analysis should have the signal type, noise type, and observing time length in the title. 

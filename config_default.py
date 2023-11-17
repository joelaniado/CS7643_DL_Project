# Use this configuration file to adjust the ML 
# scripts

# If you have multiple configurations, use the profile
# variable to keep track of them so you don't need to keep
# swiching all the variables everytime you run. Just change
# this one variable. Create a new if branch if you have
# another configuration.
profile = "full"

if profile == "test":

    ## SHARED SETTINGS
    
    home_folder = r"C:\Users\Joe Laniado\Documents\Documents\Education\Georgia_tech\dl\cs7643_project"
    
    # Only process the first n samples, or process all the samples if set to None
    number_of_samples = 10
    model_file = home_folder + "/modeloutput/model.pth"
    

    ## MAIN.PY SETTINGS
    base_source_dirpath = home_folder + "/data/nuclear/"


    ## TEST.PY SETTINGS
    images_dirpath = base_source_dirpath + "test/image"
    labels_dirpath = base_source_dirpath + "groundtruth_centerbinary_2pixelsmaller"

    device = "cpu"

    training_patience = 2

elif profile == "full":


    ## SHARED SETTINGS
    
    home_folder = r"C:\Users\Joe Laniado\Documents\Documents\Education\Georgia_tech\dl\cs7643_project"
    
    # Only process the first n samples, or process all the samples if set to None
    number_of_samples = 10
    model_file = home_folder + "/modeloutput/model.pth"
    

    ## MAIN.PY SETTINGS
    base_source_dirpath = home_folder + "/data/nuclear/"


    ## TEST.PY SETTINGS
    images_dirpath = base_source_dirpath + "test/image"
    labels_dirpath = base_source_dirpath + "test/groundtruth_centerbinary_2pixelsmaller"

    device = "cpu"

    training_patience = 2



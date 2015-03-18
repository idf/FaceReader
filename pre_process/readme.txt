* need to manually change some files in yalefaces dataset
- change the file name 'subject01.gif' to 'subject01.centerlight'
- delete the file subject01.glasses.gif as it is the same as the file subject01.glasses
- make sure there are 11 types of image files for each subject (person)
- open cmd and change directory to pre_process
- enter "python crop_face.py [your yaleface directory]"
- it will create sub folders under yalefaces eg. subject01 and save cropped image inside each respective folder